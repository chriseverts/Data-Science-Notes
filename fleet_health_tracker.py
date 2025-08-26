import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10


class FleetHealthTracker:
    """Fleet Health Tracker with Timestamp-Based Inactivity Detection
    and chart-only drop-to-zero on days with large time gaps.
    """

    def __init__(self, csv_file, dr_dates=['2025-07-15']):
        # Health scoring weights
        self.health_weights = {
            'rate_performance': 0.20,
            'engine_efficiency': 0.20,
            'power_utilization': 0.60
        }

        # Inactivity detection parameters
        # max_gap_hours controls record-level inactivity; graph_gap_hours controls drop-to-zero per-day plotting
        self.inactivity_params = {
            'max_gap_hours': 1,
            'zero_health_during_inactive': True,
            'graph_gap_hours': 24.0,          # NEW: drop-to-zero threshold for charts (hours)
            'drop_to_zero_on_gap': True       # NEW: enable chart-only zeroing behavior
        }

        # Scoring ranges
        self.engine_load_scoring = {
            'optimal_min': 0.50, 'optimal_max': 0.80, 'optimal_score': 50,
            'good_min': 0.40, 'good_max': 0.85, 'good_score': 42,
            'acceptable_min': 0.30, 'acceptable_max': 0.90, 'acceptable_score': 35,
            'fair_min': 0.20, 'fair_max': 0.95, 'fair_score': 28,
            'poor_score': 20, 'zero_score': 0
        }

        self.rpm_scoring = {
            'optimal_min': 1750, 'optimal_max': 1850, 'optimal_score': 50,
            'good_min': 1650, 'good_max': 1900, 'good_score': 42,
            'acceptable_min': 1500, 'acceptable_max': 1950, 'acceptable_score': 35,
            'fair_score': 25, 'zero_score': 15
        }

        self.gap_performance_scoring = {
            'excellent_min': 80, 'excellent_score': 100,
            'very_good_min': 70, 'very_good_score': 90,
            'good_min': 60, 'good_score': 75,
            'fair_min': 45, 'fair_score': 60,
            'poor_min': 30, 'poor_score': 40,
            'critical_score': 20
        }

        self.power_ratio_scoring = {
            'optimal_min': 0.40, 'optimal_max': 0.75, 'optimal_score': 100,
            'good_min': 0.30, 'good_max': 0.85, 'good_score': 85,
            'acceptable_min': 0.25, 'acceptable_max': 0.95, 'acceptable_score': 70,
            'fair_min': 0.15, 'fair_max': 1.10, 'fair_score': 50,
            'poor_score': 30, 'default_score': 40
        }

        # HP constraint parameters
        self.hp_constraint_params = {
            'power_efficiency': 0.8,
            'hp_leaching_constant': 100,
            'hydraulic_conversion': 40.8,
            'default_engine_load': 0.87
        }

        # Load data and initialize
        self.df = self.load_and_clean_data(csv_file)
        self.dr_dates = [pd.to_datetime(date) for date in dr_dates]

        if self.df is None:
            return

        # Handle timezone compatibility
        if 'Timestamp' in self.df.columns:
            if self.df['Timestamp'].dt.tz is not None:
                self.dr_dates = [date.tz_localize('UTC') for date in self.dr_dates]
            elif any(date.tz is not None for date in self.dr_dates):
                self.dr_dates = [date.tz_localize(None) for date in self.dr_dates]

        # Create output directory
        self.output_dir = Path("fleet_health_tracking")
        self.output_dir.mkdir(exist_ok=True)

        # Initialize tracking
        self.column_mapping = self.detect_column_mapping()
        self.stages = self._get_stages()

        print(f"Fleet Health Tracker Initialized")
        print(f"Total Records: {len(self.df):,}")
        print(f"Total Stages: {len(self.stages)}")
        print(f"Data Range: {self.df['Timestamp'].min()} to {self.df['Timestamp'].max()}")

        # Calculate health scores
        self.calculate_health_scores()

    def load_and_clean_data(self, csv_file):
        """Load and clean the data"""
        try:
            df = pd.read_csv(csv_file)
            print(f"Successfully loaded: {csv_file}")

            # Convert timestamp
            if 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])

            # Clean numeric columns
            numeric_columns = [
                'Rqstd_RPM', 'RPM', 'Pct_Eng_Load', 'Pump_Rate', 'Disch_Press',
                'HHorsepower', 'Rated_HP', 'Unit_Number'
            ]

            for col in numeric_columns:
                if col in df.columns:
                    df[col] = df[col].replace(['', ' ', 'NULL', 'null', 'N/A', 'n/a', '#N/A'], np.nan)
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Handle percentage columns
            if 'Pct_Eng_Load' in df.columns:
                df['Pct_Eng_Load'] = df['Pct_Eng_Load'].astype(str).str.replace('%', '').str.strip()
                df['Pct_Eng_Load'] = pd.to_numeric(df['Pct_Eng_Load'], errors='coerce')

            df = df.sort_values('Timestamp')
            return df

        except Exception as e:
            print(f"Error loading file: {e}")
            return None

    def _get_stages(self):
        """Get valid stages from data"""
        if 'Stage_Alias' in self.df.columns:
            self.df['Stage_Alias'] = self.df['Stage_Alias'].astype(str)
            valid_stages = self.df['Stage_Alias'].dropna()
            valid_stages = valid_stages[valid_stages != 'nan']
            valid_stages = valid_stages[valid_stages != '']
            return sorted(valid_stages.unique())
        return []

    def detect_column_mapping(self):
        """Auto-detect column names"""
        column_mapping = {}
        column_variations = {
            'rqstd_rpm': ['Rqstd_RPM', 'rqstd_rpm', 'RPM_Requested', 'RPM'],
            'pump_rate': ['Pump_Rate', 'pump_rate', 'Rate', 'rate'],
            'disch_press': ['Disch_Press', 'disch_press', 'Discharge_Pressure', 'Pressure'],
            'pct_eng_load': ['Pct_Eng_Load', 'pct_eng_load', 'Engine_Load', 'Load'],
            'rated_hp': ['Rated_HP', 'rated_hp', 'HP', 'Horsepower']
        }

        for key, variations in column_variations.items():
            for variation in variations:
                if variation in self.df.columns:
                    column_mapping[key] = variation
                    break

        return column_mapping

    def safe_float_convert(self, value, default=0.0):
        """Safely convert a value to float"""
        if value is None or pd.isna(value):
            return default

        try:
            if isinstance(value, (int, float)):
                return float(value)

            if isinstance(value, str):
                cleaned = value.strip().replace('%', '').replace(',', '')
                if cleaned == '' or cleaned.lower() in ['null', 'n/a', 'nan', 'none']:
                    return default
                return float(cleaned)

            return float(value)
        except (ValueError, TypeError):
            return default

    def safe_percentage_convert(self, value, default=0.0):
        """Safely convert percentage value"""
        converted = self.safe_float_convert(value, default)
        if converted > 1:
            return converted / 100
        return converted

    def calculate_hp_constraint_max_rate(self, row):
        """Calculate HP Constraint Max Rate"""
        try:
            disch_press_col = self.column_mapping.get('disch_press', 'Disch_Press')
            eng_load_col = self.column_mapping.get('pct_eng_load', 'Pct_Eng_Load')

            rated_hp = row.get('Rated_HP', 0)
            disch_press = row.get(disch_press_col, 0)
            engine_load_max = row.get(eng_load_col, self.hp_constraint_params['default_engine_load'])

            if isinstance(engine_load_max, str) and '%' in engine_load_max:
                engine_load_max = float(engine_load_max.replace('%', '')) / 100
            else:
                engine_load_max = float(engine_load_max)
                if engine_load_max > 1:
                    engine_load_max = engine_load_max / 100

            if any(pd.isna(x) or x <= 0 for x in [rated_hp, disch_press]):
                return np.nan

            if engine_load_max <= 0 or engine_load_max > 1:
                engine_load_max = self.hp_constraint_params['default_engine_load']

            available_hp = rated_hp * self.hp_constraint_params['power_efficiency'] * engine_load_max
            usable_hp = available_hp - self.hp_constraint_params['hp_leaching_constant']

            if usable_hp <= 0:
                return np.nan

            max_rate = self.hp_constraint_params['hydraulic_conversion'] * (usable_hp / disch_press)
            return round(max_rate, 6) if max_rate > 0 else np.nan

        except Exception:
            return np.nan

    def calculate_gap_percentage(self, row):
        """Calculate GAP% using formula"""
        try:
            pump_rate_col = self.column_mapping.get('pump_rate', 'Pump_Rate')

            current_rate = self.safe_float_convert(row.get(pump_rate_col), 0)
            potential_rate = self.safe_float_convert(row.get('HP_Constraint_Max_Rate'), 0)

            if current_rate <= 0 or potential_rate <= 0:
                return np.nan

            if current_rate > potential_rate:
                return 95.0
            else:
                gap_percentage = (current_rate / potential_rate) * 100
                return min(gap_percentage, 100.0)

        except Exception:
            return np.nan

    def detect_inactivity(self):
        """Timestamp-based inactivity detection"""
        print("Applying timestamp-based inactivity detection...")

        self.df['Is_Inactive'] = False
        self.df['Time_Gap_Hours'] = 0.0

        processed_dfs = []
        total_inactive = 0

        for stage in self.stages:
            stage_data = self.df[self.df['Stage_Alias'] == stage].copy()
            if len(stage_data) == 0:
                continue

            stage_data = stage_data.sort_values('Timestamp')

            # Calculate time gaps
            time_diffs = stage_data['Timestamp'].diff()
            stage_data['Time_Gap_Hours'] = time_diffs.dt.total_seconds() / 3600

            # Mark records after large gaps as inactive
            large_gap_mask = stage_data['Time_Gap_Hours'] > self.inactivity_params['max_gap_hours']
            stage_data.loc[large_gap_mask, 'Is_Inactive'] = True

            # First record is always active
            if len(stage_data) > 0:
                first_idx = stage_data.index[0]
                stage_data.loc[first_idx, 'Is_Inactive'] = False
                stage_data.loc[first_idx, 'Time_Gap_Hours'] = 0.0

            total_inactive += stage_data['Is_Inactive'].sum()
            processed_dfs.append(stage_data)

        # Handle records without stage aliases
        no_stage_data = self.df[~self.df['Stage_Alias'].isin(self.stages)].copy()
        if len(no_stage_data) > 0:
            no_stage_data = no_stage_data.sort_values('Timestamp')
            time_diffs = no_stage_data['Timestamp'].diff()
            no_stage_data['Time_Gap_Hours'] = time_diffs.dt.total_seconds() / 3600

            large_gap_mask = no_stage_data['Time_Gap_Hours'] > self.inactivity_params['max_gap_hours']
            no_stage_data['Is_Inactive'] = large_gap_mask

            if len(no_stage_data) > 0:
                first_idx = no_stage_data.index[0]
                no_stage_data.loc[first_idx, 'Is_Inactive'] = False

            total_inactive += no_stage_data['Is_Inactive'].sum()
            processed_dfs.append(no_stage_data)

        if processed_dfs:
            self.df = pd.concat(processed_dfs, ignore_index=True)
            self.df = self.df.sort_values('Timestamp')

        print(f"Inactive records: {total_inactive:,} ({total_inactive/len(self.df)*100:.1f}%)")

    def calculate_health_score_components(self, row):
        """Calculate health score components with inactivity detection"""

        # Check if inactive
        if row.get('Is_Inactive', False) and self.inactivity_params['zero_health_during_inactive']:
            return {
                'Rate_Performance_Score': 0.0,
                'Engine_Load_RPM_Score': 0.0,
                'Power_Utilization_Score': 0.0,
                'Total_Health_Score': 0.0
            }

        try:
            # Get column names
            pump_rate_col = self.column_mapping.get('pump_rate', 'Pump_Rate')
            disch_press_col = self.column_mapping.get('disch_press', 'Disch_Press')
            rpm_col = self.column_mapping.get('rqstd_rpm', 'Rqstd_RPM')
            eng_load_col = self.column_mapping.get('pct_eng_load', 'Pct_Eng_Load')

            # FACTOR 1: Rate Performance
            gap_percent = self.safe_float_convert(row.get('GAP%'), np.nan)

            if not pd.isna(gap_percent):
                if gap_percent >= self.gap_performance_scoring['excellent_min']:
                    performance_score = self.gap_performance_scoring['excellent_score']
                elif gap_percent >= self.gap_performance_scoring['very_good_min']:
                    performance_score = self.gap_performance_scoring['very_good_score']
                elif gap_percent >= self.gap_performance_scoring['good_min']:
                    performance_score = self.gap_performance_scoring['good_score']
                elif gap_percent >= self.gap_performance_scoring['fair_min']:
                    performance_score = self.gap_performance_scoring['fair_score']
                elif gap_percent >= self.gap_performance_scoring['poor_min']:
                    performance_score = self.gap_performance_scoring['poor_score']
                else:
                    performance_score = self.gap_performance_scoring['critical_score']
            else:
                performance_score = 0

            # FACTOR 2: Engine Load Efficiency
            engine_load_raw = row.get(eng_load_col, 55)
            engine_load = self.safe_percentage_convert(engine_load_raw, 0.55)

            if (self.engine_load_scoring['optimal_min'] <= engine_load <=
                self.engine_load_scoring['optimal_max']):
                load_score = self.engine_load_scoring['optimal_score']
            elif (self.engine_load_scoring['good_min'] <= engine_load <=
                  self.engine_load_scoring['good_max']):
                load_score = self.engine_load_scoring['good_score']
            elif (self.engine_load_scoring['acceptable_min'] <= engine_load <=
                  self.engine_load_scoring['acceptable_max']):
                load_score = self.engine_load_scoring['acceptable_score']
            elif (self.engine_load_scoring['fair_min'] <= engine_load <=
                  self.engine_load_scoring['fair_max']):
                load_score = self.engine_load_scoring['fair_score']
            elif engine_load > 0:
                load_score = self.engine_load_scoring['poor_score']
            else:
                load_score = self.engine_load_scoring['zero_score']

            # FACTOR 3: RPM Efficiency
            rqstd_rpm = self.safe_float_convert(row.get(rpm_col), 0)

            if rqstd_rpm > 0:
                if (self.rpm_scoring['optimal_min'] <= rqstd_rpm <=
                    self.rpm_scoring['optimal_max']):
                    rpm_score = self.rpm_scoring['optimal_score']
                elif (self.rpm_scoring['good_min'] <= rqstd_rpm <=
                      self.rpm_scoring['good_max']):
                    rpm_score = self.rpm_scoring['good_score']
                elif (self.rpm_scoring['acceptable_min'] <= rqstd_rpm <=
                      self.rpm_scoring['acceptable_max']):
                    rpm_score = self.rpm_scoring['acceptable_score']
                else:
                    rpm_score = self.rpm_scoring['fair_score']
            else:
                rpm_score = self.rpm_scoring['zero_score']

            engine_efficiency_score = load_score + rpm_score

            # FACTOR 4: Power Utilization
            current_rate = self.safe_float_convert(row.get(pump_rate_col), 0)
            disch_press = self.safe_float_convert(row.get(disch_press_col), 0)
            rated_hp = self.safe_float_convert(row.get('Rated_HP'), 0)

            if disch_press > 0 and current_rate > 0 and rated_hp > 0:
                hydraulic_power = (disch_press * current_rate) / self.hp_constraint_params['hydraulic_conversion']
                rate_power_ratio = hydraulic_power / rated_hp

                if (self.power_ratio_scoring['optimal_min'] <= rate_power_ratio <=
                    self.power_ratio_scoring['optimal_max']):
                    power_utilization_score = self.power_ratio_scoring['optimal_score']
                elif (self.power_ratio_scoring['good_min'] <= rate_power_ratio <=
                      self.power_ratio_scoring['good_max']):
                    power_utilization_score = self.power_ratio_scoring['good_score']
                elif (self.power_ratio_scoring['acceptable_min'] <= rate_power_ratio <=
                      self.power_ratio_scoring['acceptable_max']):
                    power_utilization_score = self.power_ratio_scoring['acceptable_score']
                elif (self.power_ratio_scoring['fair_min'] <= rate_power_ratio <=
                      self.power_ratio_scoring['fair_max']):
                    power_utilization_score = self.power_ratio_scoring['fair_score']
                else:
                    power_utilization_score = self.power_ratio_scoring['poor_score']
            else:
                power_utilization_score = self.power_ratio_scoring['default_score']

            # Calculate weighted total
            total_score = (
                performance_score * self.health_weights['rate_performance'] +
                engine_efficiency_score * self.health_weights['engine_efficiency'] +
                power_utilization_score * self.health_weights['power_utilization']
            )

            total_score = min(100, max(0, total_score))

            return {
                'Rate_Performance_Score': round(performance_score, 1),
                'Engine_Load_RPM_Score': round(engine_efficiency_score, 1),
                'Power_Utilization_Score': round(power_utilization_score, 1),
                'Total_Health_Score': round(total_score, 1)
            }

        except Exception as e:
            return {
                'Rate_Performance_Score': np.nan,
                'Engine_Load_RPM_Score': np.nan,
                'Power_Utilization_Score': np.nan,
                'Total_Health_Score': np.nan
            }

    def calculate_health_scores(self):
        """Calculate health scores with inactivity detection"""
        print("Calculating health scores...")

        # Calculate HP Constraint Max Rate and GAP%
        self.df['HP_Constraint_Max_Rate'] = self.df.apply(self.calculate_hp_constraint_max_rate, axis=1)
        self.df['GAP%'] = self.df.apply(self.calculate_gap_percentage, axis=1)

        # Apply inactivity detection
        self.detect_inactivity()

        # Calculate health score components
        health_scores = self.df.apply(self.calculate_health_score_components, axis=1)

        for component in ['Rate_Performance_Score', 'Engine_Load_RPM_Score',
                         'Power_Utilization_Score', 'Total_Health_Score']:
            self.df[component] = [score[component] for score in health_scores]

        # Add health grades and status
        def get_health_grade(score):
            if pd.isna(score) or score == 0:
                return 'F' if score == 0 else 'N/A'
            elif score >= 90: return 'A+'
            elif score >= 85: return 'A'
            elif score >= 80: return 'A-'
            elif score >= 75: return 'B+'
            elif score >= 70: return 'B'
            elif score >= 65: return 'B-'
            elif score >= 60: return 'C+'
            elif score >= 55: return 'C'
            elif score >= 50: return 'C-'
            elif score >= 45: return 'D+'
            elif score >= 40: return 'D'
            else: return 'F'

        def get_health_status(score):
            if pd.isna(score): return 'Unknown'
            elif score == 0: return 'Inactive'
            elif score >= 90: return 'Excellent'
            elif score >= 80: return 'Very Good'
            elif score >= 70: return 'Good'
            elif score >= 60: return 'Fair'
            elif score >= 50: return 'Poor'
            else: return 'Critical'

        self.df['Health_Grade'] = self.df['Total_Health_Score'].apply(get_health_grade)
        self.df['Health_Status'] = self.df['Total_Health_Score'].apply(get_health_status)

        print(f"Health scores calculated. Average: {self.df['Total_Health_Score'].mean():.1f}")

    def create_health_trend_line_graph(self):
        """Create line graph showing health trends with per-day drop-to-zero on large gaps"""
        print("Creating health trend line graph...")

        fig, ax = plt.subplots(1, 1, figsize=(20, 8))

        # Calculate daily fleet averages
        daily = self.df.copy()
        daily['Date'] = daily['Timestamp'].dt.date

        daily_health = daily.groupby('Date').agg({
            'Total_Health_Score': 'mean',
            'Is_Inactive': 'mean'
        }).dropna()

        if len(daily_health) == 0:
            print("No daily health data to plot")
            return

        # Detect large-gap days and zero-out for plotting if enabled
        if 'Time_Gap_Hours' in daily.columns:
            gap_by_day = daily.groupby('Date')['Time_Gap_Hours'].max().fillna(0)
        else:
            gap_by_day = pd.Series(0, index=daily_health.index)

        large_gap_threshold = float(self.inactivity_params.get('graph_gap_hours', 24.0))
        daily_health['Has_Large_Gap'] = (gap_by_day.reindex(daily_health.index).fillna(0) >= large_gap_threshold)

        if self.inactivity_params.get('drop_to_zero_on_gap', True):
            daily_health.loc[daily_health['Has_Large_Gap'], 'Total_Health_Score'] = 0.0

        # Plot health trend
        ax.plot(daily_health.index, daily_health['Total_Health_Score'],
               linewidth=3, color='blue', alpha=0.8, label='Fleet Average Health',
               marker='o', markersize=3)

        # Highlight days with inactivity or large gaps
        inactive_days = daily_health[(daily_health['Is_Inactive'] > 0.1) | (daily_health['Has_Large_Gap'])]
        if len(inactive_days) > 0:
            ax.scatter(inactive_days.index, inactive_days['Total_Health_Score'],
                      color='red', s=50, alpha=0.7, label='Days with Inactivity/Large Gaps', zorder=5)

        # Add DR date lines
        for i, dr_date in enumerate(self.dr_dates):
            dr_date_for_plot = dr_date
            if dr_date.tz is not None:
                dr_date_for_plot = dr_date.tz_convert(None)

            label = f'DR {i+1}: {dr_date_for_plot.strftime("%m/%d")}' if len(self.dr_dates) > 1 else f'DR: {dr_date_for_plot.strftime("%m/%d")}'
            ax.axvline(dr_date_for_plot.date(), color='black', linestyle='--',
                      linewidth=3, label=label, alpha=0.7)

        # Formatting
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Health Score', fontsize=12)
        ax.set_title('FLEET HEALTH TREND WITH INACTIVITY DETECTION',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 100)

        # Add summary
        avg_health = daily_health['Total_Health_Score'].mean()
        total_inactive_days = len(inactive_days)

        summary_text = f"""FLEET HEALTH SUMMARY
Average Health: {avg_health:.1f}
Days with Inactivity/Large Gaps: {total_inactive_days}
Record Gap Threshold: {self.inactivity_params['max_gap_hours']}h
Drop-to-Zero Gap (Chart): {self.inactivity_params.get('graph_gap_hours', 24.0)}h"""

        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
               bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8),
               verticalalignment='top', fontsize=10)

        plt.tight_layout()

        # Save chart
        chart_path = self.output_dir / f"fleet_health_trend_DR{len(self.dr_dates)}.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"Health trend chart saved: {chart_path}")
        plt.show()

        return chart_path

    def create_inactivity_dashboard(self):
        """Create inactivity analysis dashboard with large-gap day visualization"""
        print("Creating inactivity dashboard...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('INACTIVITY DETECTION ANALYSIS', fontsize=16, fontweight='bold')

        # 1. Health trend with inactivity markers
        ax1 = axes[0, 0]
        daily = self.df.copy()
        daily['Date'] = daily['Timestamp'].dt.date

        daily_health = daily.groupby('Date').agg({
            'Total_Health_Score': 'mean',
            'Is_Inactive': 'sum'
        })

        if 'Time_Gap_Hours' in daily.columns:
            gap_by_day = daily.groupby('Date')['Time_Gap_Hours'].max().fillna(0)
        else:
            gap_by_day = pd.Series(0, index=daily_health.index)

        large_gap_threshold = float(self.inactivity_params.get('graph_gap_hours', 24.0))
        daily_health['Has_Large_Gap'] = (gap_by_day.reindex(daily_health.index).fillna(0) >= large_gap_threshold)

        if self.inactivity_params.get('drop_to_zero_on_gap', True):
            daily_health.loc[daily_health['Has_Large_Gap'], 'Total_Health_Score'] = 0.0

        ax1.plot(daily_health.index, daily_health['Total_Health_Score'],
                linewidth=2, color='blue', alpha=0.8, label='Fleet Health')

        inactive_days = daily_health[(daily_health['Is_Inactive'] > 0) | (daily_health['Has_Large_Gap'])]
        if len(inactive_days) > 0:
            ax1.scatter(inactive_days.index, inactive_days['Total_Health_Score'],
                       color='red', s=30, alpha=0.7, label='Days with Inactivity/Large Gaps')

        ax1.set_title('Health Trend with Inactivity/Large-Gap Markers')
        ax1.set_ylabel('Health Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        # 2. Time gap distribution
        ax2 = axes[0, 1]
        if 'Time_Gap_Hours' in self.df.columns:
            gap_data = self.df[self.df['Time_Gap_Hours'] > 0]['Time_Gap_Hours']
            if len(gap_data) > 0:
                ax2.hist(gap_data, bins=30, alpha=0.7, color='orange', edgecolor='black')
                ax2.axvline(self.inactivity_params['max_gap_hours'], color='red',
                           linestyle='--', linewidth=2, label=f"Record Threshold: {self.inactivity_params['max_gap_hours']}h")
                ax2.axvline(self.inactivity_params.get('graph_gap_hours', 24.0), color='purple',
                           linestyle='--', linewidth=2, label=f"Chart Zero Threshold: {self.inactivity_params.get('graph_gap_hours', 24.0)}h")
                ax2.set_xlabel('Time Gap (Hours)')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Distribution of Time Gaps')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

        # 3. Activity status pie chart
        ax3 = axes[1, 0]
        activity_counts = self.df['Is_Inactive'].value_counts()
        labels = ['Active', 'Inactive']
        colors = ['green', 'red']
        values = [activity_counts.get(False, 0), activity_counts.get(True, 0)]

        ax3.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Fleet Activity Distribution')

        # 4. Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')

        total_records = len(self.df)
        inactive_records = len(self.df[self.df['Is_Inactive'] == True])
        zero_health_records = len(self.df[self.df['Total_Health_Score'] == 0])
        avg_health = self.df['Total_Health_Score'].mean()

        # Count days with large gaps for dashboard summary
        if 'Time_Gap_Hours' in self.df.columns:
            tmp_daily = self.df.copy()
            tmp_daily['Date'] = tmp_daily['Timestamp'].dt.date
            lg_threshold = float(self.inactivity_params.get('graph_gap_hours', 24.0))
            days_with_large_gaps = (
                tmp_daily.groupby('Date')['Time_Gap_Hours'].max().fillna(0) >= lg_threshold
            ).sum()
        else:
            days_with_large_gaps = 0

        summary_text = f"""INACTIVITY SUMMARY
========================

Detection Method: Time Gaps
Record Threshold: {self.inactivity_params['max_gap_hours']} hours
Chart Drop-to-Zero Threshold: {self.inactivity_params.get('graph_gap_hours', 24.0)} hours

Statistics:
• Total Records: {total_records:,}
• Inactive Records: {inactive_records:,} ({inactive_records/total_records*100:.1f}%)
• Zero Health Records: {zero_health_records:,}
• Average Health: {avg_health:.1f}
• Days with ≥ Chart Threshold Gap: {int(days_with_large_gaps)}
"""

        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))

        plt.tight_layout()

        # Save dashboard
        dashboard_path = self.output_dir / f"inactivity_dashboard_DR{len(self.dr_dates)}.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        print(f"Inactivity dashboard saved: {dashboard_path}")
        plt.show()

        return dashboard_path

    def export_data(self):
        """Export enhanced data"""
        print("Exporting enhanced data...")

        export_df = self.df.copy()
        export_df = export_df.sort_values(['Stage_Alias', 'Timestamp'])

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"fleet_health_data_DR{len(self.dr_dates)}_{timestamp}.csv"
        output_path = self.output_dir / output_filename

        export_df.to_csv(output_path, index=False)

        print(f"Data exported: {output_path}")
        print(f"Records exported: {len(export_df):,}")

        return output_path

    def run_analysis(self):
        """Run the complete analysis"""
        print("Running Fleet Health Analysis")
        print("=" * 50)

        # Show data summary
        print(f"Total records: {len(self.df):,}")
        print(f"Date range: {self.df['Timestamp'].min()} to {self.df['Timestamp'].max()}")
        print(f"Stages found: {len(self.stages)}")

        # Show health breakdown
        if 'Is_Inactive' in self.df.columns:
            inactive_count = len(self.df[self.df['Is_Inactive'] == True])
            active_count = len(self.df[self.df['Is_Inactive'] == False])
            zero_health_count = len(self.df[self.df['Total_Health_Score'] == 0])

            print(f"\nHealth Score Breakdown:")
            print(f"Inactive records: {inactive_count:,} ({inactive_count/len(self.df)*100:.1f}%)")
            print(f"Active records: {active_count:,} ({active_count/len(self.df)*100:.1f}%)")
            print(f"Zero health scores: {zero_health_count:,} ({zero_health_count/len(self.df)*100:.1f}%)")
            print(f"Average health score: {self.df['Total_Health_Score'].mean():.1f}")

            # Show grade distribution
            health_grades = self.df['Health_Grade'].value_counts()
            print(f"\nGrade Distribution:")
            for grade in ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'F']:
                if grade in health_grades:
                    count = health_grades[grade]
                    print(f"{grade}: {count:,} ({count/len(self.df)*100:.1f}%)")

        # Create visualizations
        print("\nCreating visualizations...")
        chart_path = self.create_health_trend_line_graph()
        dashboard_path = self.create_inactivity_dashboard()

        # Export data
        export_path = self.export_data()

        print(f"\nAnalysis Complete!")
        print(f"Results saved to:")
        print(f"  Data: {export_path}")
        print(f"  Chart: {chart_path}")
        print(f"  Dashboard: {dashboard_path}")

        return {
            'data': export_path,
            'chart': chart_path,
            'dashboard': dashboard_path
        }


def main():
    """Main execution function"""

    print("Fleet Health Tracker - Timestamp-Based Inactivity Detection")
    print("=" * 60)

    # Get input file
    csv_file = input("Enter CSV file path (or press Enter for 'analyzed_pump_data_620295.csv'): ").strip()
    if not csv_file:
        csv_file = 'analyzed_pump_data_620295.csv'

    # Get DR dates
    print("\nEnter DR (Deficiency Report) dates:")
    print("Format: YYYY-MM-DD (press Enter after each date, empty line to finish)")
    print("Default: 2025-07-15")

    dr_dates = []
    while True:
        dr_date = input(f"DR Date {len(dr_dates)+1} (or press Enter to finish): ").strip()
        if not dr_date:
            if not dr_dates:
                dr_dates = ['2025-07-15']
            break
        try:
            pd.to_datetime(dr_date)
            dr_dates.append(dr_date)
        except:
            print("Invalid date format. Please use YYYY-MM-DD")

    try:
        # Initialize and run analysis
        tracker = FleetHealthTracker(csv_file, dr_dates)

        if tracker.df is None:
            print("Failed to load data. Exiting...")
            return

        results = tracker.run_analysis()

        print(f"\nAnalysis complete! Files generated:")
        for result_type, path in results.items():
            print(f"  {result_type}: {path}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

