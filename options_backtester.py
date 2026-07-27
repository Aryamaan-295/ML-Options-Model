import os
import warnings
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# Suppress warnings for cleaner execution
warnings.filterwarnings('ignore')

class OptionsBacktester:
    """
    A robust backtesting engine for options strategies.
    Loads data once upon initialization for efficient multiple-date querying.
    """
    
    def __init__(self, data_path: str = None, default_lot_size: int = 50):
        from dotenv import load_dotenv
        load_dotenv()
        
        self.data_path = data_path or os.getenv("OPTIONS_DATA_FILE", "data/NIFTY_opt_final.csv")
        self.default_lot_size = default_lot_size
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
            
        print(f"Initializing Backtester... Loading Data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        self._clean_data()
        print("Data successfully loaded and cleaned.")

    def _clean_data(self):
        """Standardizes columns, date formats, and data types safely."""
        self.df.columns = self.df.columns.str.strip()
        col_map = {c.lower().replace(' ', ''): c for c in self.df.columns}
        
        # Standardize column names dynamically
        rename_dict = {}
        for target, aliases in {
            'Expiry': ['expiry'], 'LotSize': ['lotsize'], 
            'Strike': ['strike'], 'Type': ['type', 'optiontype'], 
            'Volume': ['volume'], 'Datetime': ['datetime', 'date']
        }.items():
            for alias in aliases:
                if alias in col_map:
                    rename_dict[col_map[alias]] = target
                    break
        self.df.rename(columns=rename_dict, inplace=True)

        # STRICT DATE NORMALIZATION: Convert everything to pandas datetime64[ns] to prevent mismatch bugs
        self.df['Datetime'] = pd.to_datetime(self.df['Datetime'])
        self.df['Date'] = self.df['Datetime'].dt.normalize() 
        self.df['Expiry'] = pd.to_datetime(self.df['Expiry']).dt.normalize()
        
        # Handle Strikes
        if 'Strike' in self.df.columns:
            self.df['Strike'] = pd.to_numeric(self.df['Strike'], errors='coerce').astype(float)
            
        # Safely map Type back to CE/PE in memory
        if 'Type' in self.df.columns:
            self.df['Type'] = self.df['Type'].astype(str).str.upper().str.strip()
            self.df.loc[self.df['Type'] == 'C', 'Type'] = 'CE'
            self.df.loc[self.df['Type'] == 'P', 'Type'] = 'PE'

    @staticmethod
    def _black_scholes_price(S, K, T, r, sigma, opt_type='CE'):
        if isinstance(T, float): 
            if T <= 0: 
                return np.maximum(0, S - K) if opt_type == 'CE' else np.maximum(0, K - S)
        else: 
            T = np.maximum(T, 1e-9)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if opt_type == 'CE':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return price

    def _get_leg_data(self, leg_config, df_day):
        """Fetches row data strictly from the pre-filtered df_day and validates existence/liquidity."""
        strike = float(leg_config['strike']) 
        
        opt_type = str(leg_config['type']).upper().strip()
        if opt_type == 'C': opt_type = 'CE'
        elif opt_type == 'P': opt_type = 'PE'
        
        expiry = pd.to_datetime(leg_config['expiry']).normalize()

        # Query directly against the day's dataframe
        mask = (df_day['Expiry'] == expiry) & \
               (df_day['Strike'] == strike) & \
               (df_day['Type'] == opt_type)
        
        row = df_day[mask]
        
        if row.empty:
            return None, "missing"
        
        row = row.iloc[0]
        
        # Illiquidity Check
        if 'Volume' in row.index:
            if pd.isna(row['Volume']) or float(row['Volume']) <= 0.0:
                return None, "illiquid"

        span_risk = [float(row.get(f's_{i}', 0.0)) for i in range(1, 17)]
        leg_lot_size = int(row.get('LotSize', self.default_lot_size))

        return {
            'strike': strike,
            'type': opt_type,
            'action': leg_config['action'].lower(),
            'qty': leg_config.get('qty', 1),
            'expiry': expiry,
            'entry_price': row['Close'],
            'lot_size': leg_lot_size,
            'span_risk': span_risk, 
            'greeks': {
                'delta': row.get('delta', 0),
                'gamma': row.get('gamma', 0),
                'theta': row.get('theta', 0),
                'vega': row.get('vega', 0)
            }
        }, "ok"

    def _calculate_margin(self, parsed_legs, spot_price, analysis_date_ts):
        """
        Calculates accurate exchange margin matching Indian Index constraints.
        Exposure margin is calculated based on theoretical Futures Price.
        """
        combined_span_risk = np.zeros(16)
        total_exposure = 0.0
        total_sell_lots = 0

        # Estimate Futures price via Cost of Carry (6% Risk Free Rate)
        days_to_expiry = (parsed_legs[0]['expiry'].normalize() - analysis_date_ts).days if parsed_legs else 0
        t_years = max(days_to_expiry / 365.0, 0)
        futures_price = spot_price * np.exp(0.06 * t_years)

        for leg in parsed_legs:
            qty_lots = leg['qty']
            leg_lot_size = leg['lot_size']
            risk_arr = np.array(leg['span_risk'])
            risk_vec = risk_arr * qty_lots * leg_lot_size
            
            if leg['action'] == 'sell':
                combined_span_risk -= risk_vec 
                # Strict 2% exposure for Nifty Index Options based on FUTURES notional
                total_exposure += (futures_price * leg_lot_size * qty_lots) * 0.02
                total_sell_lots += qty_lots
            else:
                combined_span_risk += risk_vec 

        if total_sell_lots == 0 and np.all(combined_span_risk == 0):
            return 0.0, 0.0, 0.0

        # SPAN margin derived natively from the 16 arrays.
        span_margin = max(0.0, np.max(combined_span_risk))
        return span_margin + total_exposure, span_margin, total_exposure

    def calculate_basket_payoff(self, portfolio: list, analysis_date: str, target_pnl_date: str = None, show_plot: bool = True, verbose: bool = True):
        # 1. Parse Dates safely to datetime64[ns]
        analysis_date_ts = pd.to_datetime(analysis_date).normalize()
        df_day = self.df[self.df['Date'] == analysis_date_ts].copy()
        
        if df_day.empty:
            return {"status": "error", "messages": [f"No spot/underlying data found in dataset for Analysis Date: {analysis_date_ts.date()}"]}

        # Safety fallback for Spot Price
        spot_price = df_day['SpotPrice'].iloc[0] if 'SpotPrice' in df_day.columns else df_day['Close'].iloc[0]
        
        # 2. Parse Legs & Collect Errors
        legs = []
        missing_legs = []
        illiquid_legs = []

        for l in portfolio:
            leg_data, status = self._get_leg_data(l, df_day)
            if status == "missing":
                missing_legs.append(l)
            elif status == "illiquid":
                illiquid_legs.append(l)
            else:
                legs.append(leg_data)

        # 3. Handle Errors/Issues with SMART DIAGNOSTICS
        if missing_legs:
            avail_expiries = [d.date() for d in pd.Series(df_day['Expiry'].unique()).dropna()]
            msgs = ["The following requested options contracts are missing from the dataset:"]
            for ml in missing_legs:
                req_exp_ts = pd.to_datetime(ml['expiry']).normalize()
                msgs.append(f"  • {ml['action'].upper()} {ml.get('qty',1)}x {ml['strike']} {ml['type']} | Expiry: {req_exp_ts.date()}")
                if req_exp_ts.date() not in avail_expiries:
                    sample_exp = [str(x) for x in avail_expiries[:4]]
                    msgs.append(f"      -> Reason: Expiry {req_exp_ts.date()} not found on {analysis_date_ts.date()}. Available: {sample_exp}...")
                else:
                    avail_strikes = sorted(df_day[df_day['Expiry'] == req_exp_ts]['Strike'].unique())
                    req_strike = float(ml['strike'])
                    if req_strike not in avail_strikes:
                        msgs.append(f"      -> Reason: Strike {req_strike} not traded for this expiry. Available range: {avail_strikes[0]} to {avail_strikes[-1]}")
                    else:
                        msgs.append(f"      -> Reason: Strike & Expiry exist, but Type '{ml['type']}' is missing.")
            return {"status": "error", "messages": msgs}

        if illiquid_legs:
            msgs = ["Trade execution halted due to illiquidity (Volume = 0) in the following legs:"]
            for il in illiquid_legs:
                req_exp_ts = pd.to_datetime(il['expiry']).normalize()
                msgs.append(f"  • {il['action'].upper()} {il.get('qty',1)}x {il['strike']} {il['type']} | Expiry: {req_exp_ts.date()}")
            return {"status": "issue", "messages": msgs}

        # 4. Proceed with Calculations
        earliest_expiry = min([l['expiry'] for l in legs]).normalize()
        target_date = pd.to_datetime(target_pnl_date).normalize() if target_pnl_date else earliest_expiry

        range_min = spot_price - 2000
        range_max = spot_price + 2000
        spots = np.linspace(range_min, range_max, 400)
        
        # INJECT EXACT STRIKES: This completely fixes the "Clipping Bug" allowing np.max/min to find absolute limits
        leg_strikes = [l['strike'] for l in legs]
        spots = np.sort(np.unique(np.append(spots, leg_strikes)))
        
        payoff_expiry = np.zeros_like(spots)
        payoff_target = np.zeros_like(spots)
        net_greeks = {'delta':0, 'gamma':0, 'theta':0, 'vega':0}
        total_premium = 0
        
        if verbose:
            print(f"\n--- Strategy Breakdown | Analysis Day: {analysis_date_ts.date()} | Spot: {spot_price} ---")

        for leg in legs:
            direction = -1 if leg['action'] == 'sell' else 1
            
            # Accurate computation of Net Premium stored per-leg
            premium = leg['entry_price'] * leg['qty'] * leg['lot_size']
            leg['cash_premium'] = premium 
            total_premium += premium if leg['action'] == 'sell' else -premium

            cash_multiplier = leg['qty'] * leg['lot_size'] * direction
            greek_multiplier = leg['qty'] * direction

            for k in net_greeks:
                net_greeks[k] += leg['greeks'].get(k, 0) * greek_multiplier

            days_from_analysis_to_leg = (leg['expiry'] - earliest_expiry).days
            T_years_forward = days_from_analysis_to_leg / 365.0
            
            val_at_expiry = self._black_scholes_price(spots, leg['strike'], T_years_forward, 0.06, 0.2, leg['type']) if T_years_forward > 0 else np.maximum(spots - leg['strike'], 0) if leg['type'] == 'CE' else np.maximum(leg['strike'] - spots, 0)
            payoff_expiry += (val_at_expiry - leg['entry_price']) * cash_multiplier
            
            days_to_target = (leg['expiry'] - target_date).days
            T_years_target = days_to_target / 365.0
            
            val_at_target = self._black_scholes_price(spots, leg['strike'], T_years_target, 0.06, 0.2, leg['type']) if T_years_target > 0 else np.maximum(spots - leg['strike'], 0) if leg['type'] == 'CE' else np.maximum(leg['strike'] - spots, 0)
            payoff_target += (val_at_target - leg['entry_price']) * cash_multiplier
            
            if verbose:
                print(f"{leg['action'].upper()} {leg['strike']} {leg['type']} ({leg['expiry'].date()}) @ ₹{leg['entry_price']:.2f} | Qty: {leg['qty']} | Lot: {leg['lot_size']}")

        # 5. Compile Metrics
        max_profit = np.max(payoff_expiry)
        max_loss = np.min(payoff_expiry)
        
        if (payoff_expiry[-1] > payoff_expiry[-2] + 10) or (payoff_expiry[0] > payoff_expiry[1] + 10): max_profit = np.inf
        if (payoff_expiry[-1] < payoff_expiry[-2] - 10) or (payoff_expiry[0] < payoff_expiry[1] - 10): max_loss = -np.inf
        
        rr = abs(max_loss / max_profit) if (max_profit > 0 and max_profit != np.inf) else 0
        
        breakevens = []
        for i in range(len(payoff_expiry) - 1):
            if np.sign(payoff_expiry[i]) != np.sign(payoff_expiry[i+1]):
                x1, x2 = spots[i], spots[i+1]
                y1, y2 = payoff_expiry[i], payoff_expiry[i+1]
                zero_x = x1 - y1 * ((x2 - x1) / (y2 - y1))
                breakevens.append(zero_x)

        margin, span_part, exp_part = self._calculate_margin(legs, spot_price, analysis_date_ts)
        roi = (max_profit / margin * 100) if (margin > 0 and max_profit != np.inf) else 0
        
        metrics = {
            'Spot Price': spot_price, 'Max Profit': max_profit, 'Max Loss': max_loss,
            'RR': rr, 'Breakevens': sorted(breakevens), 'Margin Req': margin,
            'Span Margin': span_part, 'Exposure Margin': exp_part, 'ROI (%)': roi,
            'Total Premium': total_premium, 'Credit/Debit': "Credit" if total_premium > 0 else "Debit"
        }

        if show_plot:
            self._plot_payoff(spots, payoff_expiry, payoff_target, spot_price, target_date.date(), metrics)

        # Added "legs" export
        return {"status": "success", "metrics": metrics, "legs": legs, "greeks": net_greeks, "arrays": {"spots": spots, "payoff_expiry": payoff_expiry, "payoff_target": payoff_target}}

    def _plot_payoff(self, spots, payoff_expiry, payoff_target, spot_price, target_date, metrics):
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=spots, y=np.maximum(payoff_expiry, 0), mode='lines', name='Profit', line=dict(color='green', width=0), fill='tozeroy', fillcolor='rgba(0, 180, 0, 0.2)', hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=spots, y=np.minimum(payoff_expiry, 0), mode='lines', name='Loss', line=dict(color='red', width=0), fill='tozeroy', fillcolor='rgba(200, 0, 0, 0.2)', hoverinfo='skip'))
        
        fig.add_trace(go.Scatter(x=spots, y=payoff_expiry, mode='lines', name='PnL @ Expiry', line=dict(color='black', width=2), hovertemplate='Spot: %{x:.0f}<br>Expiry PnL: ₹%{y:,.0f}<extra></extra>'))
        fig.add_trace(go.Scatter(x=spots, y=payoff_target, mode='lines', name=f'PnL @ {target_date}', line=dict(color='blue', width=2, dash='dash'), hovertemplate=f'Spot: %{{x:.0f}}<br>{target_date} PnL: ₹%{{y:,.0f}}<extra></extra>'))
        
        fig.add_vline(x=spot_price, line_dash="dash", line_color="orange", annotation_text="Spot")
        fig.add_hline(y=0, line_color="gray", line_width=1)
        
        max_p = metrics['Max Profit'] if metrics['Max Profit'] != np.inf else 20000
        max_l = metrics['Max Loss'] if metrics['Max Loss'] != -np.inf else -20000

        y_max = max(max_p, 0)
        y_min = min(max_l, 0)

        padding = (y_max - y_min) * 0.2
        if padding == 0: 
            padding = 2000 

        disp_max = y_max + padding
        disp_min = y_min - padding
        
        fig.update_yaxes(range=[disp_min, disp_max])
        
        fig.update_layout(title="Strategy Payoff Diagram", xaxis_title="Index Price", yaxis_title="P&L (₹)", template="plotly_white", height=600, width=1000, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.show()