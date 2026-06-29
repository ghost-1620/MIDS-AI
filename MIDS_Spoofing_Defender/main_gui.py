import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import math
import random

# Import your working backend pipeline functions
from src.data_simulator import generate_synthetic_marinecadastre
from src.attack_injector import inject_gradual_drift
from src.kinematics_engine import calculate_kinematics
from src.model_engine import train_gru_model
import torch
import numpy as np


class MIDSDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("MIDS - Tactical Maritime AI Command & Intrusion Dashboard v5.0 Master")
        self.root.geometry("1550x950")  # Expanded resolution for big-screen presentation
        self.root.configure(bg="#020408")

        self.trained_model = None
        self.features_list = ['V_calc', 'Speed_Delta', 'Angular_Velocity']

        # --- GLOBAL LOGISTICAL METRICS ---
        self.current_leg = "LEG 1: Karachi ➔ Muscat"
        self.total_containers = 4500
        self.base_vessel_weight = 35000
        self.current_cargo_weight = self.total_containers * 14.2
        self.weather_condition = "CLEAR / CALM SEAS"

        self.nearby_ships = [
            [45, 0.4, "TANKER_AL_SAD", "VLCC Tanker"],
            [120, 0.7, "CARGO_PACIFIC", "Container Ship"],
            [280, 0.3, "NAVY_FRIGATE", "Military Guard"],
            [210, 0.8, "FISHING_TRAWLER", "Trawler"]
        ]

        self.pulse_r = 0
        self.pulse_dir = 1

        self.configure_styles()
        self.build_ui()
        self.init_radar_canvas()

    def configure_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("Treeview", background="#060913", foreground="#E2E8F0", fieldbackground="#060913",
                             rowheight=24, font=("Consolas", 9))
        self.style.configure("Treeview.Heading", background="#1E1B4B", foreground="#22D3EE",
                             font=("Consolas", 9, "bold"), borderwidth=1, relief="flat")
        self.style.map("Treeview", background=[('selected', '#4338CA')], foreground=[('selected', '#FFFFFF')])

    def build_ui(self):
        # --- TOP NAVIGATION BANNER ---
        title_frame = tk.Frame(self.root, bg="#0A0F1D", height=75, bd=0)
        title_frame.pack(fill="x", side="top", padx=0, pady=0)

        accent_line = tk.Frame(self.root, bg="#22D3EE", height=2)
        accent_line.pack(fill="x", side="top")

        title_lbl = tk.Label(title_frame, text=" ⚓  MIDS-AI : CYBER MARITIME TACTICAL COMMAND & INSTRUMENTATION",
                             fg="#22D3EE", bg="#0A0F1D", font=("Consolas", 15, "bold"))
        title_lbl.pack(pady=22, padx=25, side="left")

        self.status_indicator = tk.Label(title_frame, text="SYSTEM CORE ONLINE", fg="#00FF66", bg="#111827",
                                         font=("Consolas", 10, "bold"), padx=16, pady=8, bd=1, relief="solid")
        self.status_indicator.pack(pady=15, padx=25, side="right")

        # --- MAIN SPLIT CONTAINER ---
        main_container = tk.Frame(self.root, bg="#020408")
        main_container.pack(fill="both", expand=True, padx=20, pady=15)

        # Left Panel (Controls & Telemetry)
        left_panel = tk.Frame(main_container, bg="#0A0F1D", width=360, bd=0)
        left_panel.pack(fill="y", side="left", padx=(0, 15))
        left_panel.pack_propagate(False)

        # Right Panel (Expanded Scope Workspace)
        right_panel = tk.Frame(main_container, bg="#020408")
        right_panel.pack(fill="both", expand=True, side="right")

        # --- LEFT PANEL LAYOUT ---
        tk.Label(left_panel, text="OPERATIONAL COMMAND UNIT", fg="#64748B", bg="#0A0F1D",
                 font=("Consolas", 10, "bold")).pack(pady=(15, 10))

        self.btn_train = tk.Button(left_panel, text="🎛️  DEPLOY AI ENGINES", command=self.start_ai_training,
                                   bg="#4338CA", fg="white", font=("Consolas", 9, "bold"), bd=0, cursor="hand2",
                                   height=2)
        self.btn_train.pack(fill="x", padx=20, pady=6)

        self.btn_simulate = tk.Button(left_panel, text="📡  ENGAGE TAC-RADAR", command=self.run_live_simulation,
                                      bg="#1E293B", fg="#475569", font=("Consolas", 9, "bold"), bd=0, state="disabled",
                                      cursor="hand2", height=2)
        self.btn_simulate.pack(fill="x", padx=20, pady=6)

        def create_card(parent, title, fg_color):
            frame = tk.LabelFrame(parent, text=f" {title} ", fg=fg_color, bg="#111827", font=("Consolas", 9, "bold"),
                                  padx=12, pady=10, bd=1, relief="solid")
            frame.pack(fill="x", padx=20, pady=8)
            return frame

        cargo_frame = create_card(left_panel, "MANIFEST DATABASE", "#38BDF8")
        self.lbl_leg = tk.Label(cargo_frame, text=f"Route: {self.current_leg}", fg="#F1F5F9", bg="#111827",
                                font=("Consolas", 9, "bold"), justify="left", anchor="w")
        self.lbl_leg.pack(fill="x")
        self.lbl_manifest = tk.Label(cargo_frame, text=f"Containers: {self.total_containers} TEU", fg="#94A3B8",
                                     bg="#111827", font=("Consolas", 9), justify="left", anchor="w")
        self.lbl_manifest.pack(fill="x", pady=(4, 0))

        fuel_frame = create_card(left_panel, "TELEMETRY MONITOR", "#EC4899")
        self.lbl_fuel_rate = tk.Label(fuel_frame, text="Fuel Consumption: 0.00 Tons/Hr", fg="#F1F5F9", bg="#111827",
                                      font=("Consolas", 9), justify="left", anchor="w")
        self.lbl_fuel_rate.pack(fill="x")
        self.lbl_zonation = tk.Label(fuel_frame, text="Zone: Initializing...", fg="#A855F7", bg="#111827",
                                     font=("Consolas", 9, "bold"), justify="left", anchor="w")
        self.lbl_zonation.pack(fill="x", pady=(4, 0))

        weather_frame = create_card(left_panel, "METEOROLOGICAL HUB", "#00FF66")
        self.lbl_weather = tk.Label(weather_frame, text=f"Weather: {self.weather_condition}", fg="#F1F5F9",
                                    bg="#111827", font=("Consolas", 9), justify="left", anchor="w")
        self.lbl_weather.pack(fill="x")
        self.lbl_weather_action = tk.Label(weather_frame, text="PATH VECTOR: PROCEED", fg="#00FF66", bg="#111827",
                                           font=("Consolas", 9, "bold"), justify="left", anchor="w")
        self.lbl_weather_action.pack(fill="x", pady=(4, 0))

        self.log_box = tk.Text(left_panel, bg="#010204", fg="#34D399", font=("Consolas", 8), bd=1, relief="solid",
                               wrap="word", highlightthickness=0)
        self.log_box.pack(fill="both", expand=True, padx=20, pady=(10, 15))

        # --- RIGHT PANEL LAYOUT (CRITICAL TELEMETRY LIST) ---
        table_frame = tk.Frame(right_panel, bg="#0A0F1D", bd=0)
        table_frame.pack(fill="x", side="top", pady=(0, 15))
        table_frame.pack_propagate(False)
        table_frame.config(height=160)  # Scaled down slightly to yield massive space to the canvas below

        columns = ('time', 'v_calc', 'sog', 'delta', 'prediction', 'status')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings')
        self.tree.heading('time', text='TIMESTAMP')
        self.tree.heading('v_calc', text='V_CALC')
        self.tree.heading('sog', text='SOG INPUT')
        self.tree.heading('delta', text='DELTA')
        self.tree.heading('prediction', text='AI PROB')
        self.tree.heading('status', text='EVALUATION')
        for col in columns: self.tree.column(col, width=115, anchor="center")
        self.tree.pack(fill="both", expand=True, padx=15, pady=12)

        # --- THE MASSIVE RADAR FRAME ---
        self.radar_frame = tk.Frame(right_panel, bg="#0A0F1D")
        self.radar_frame.pack(fill="both", expand=True, side="bottom")

        legend_frame = tk.Frame(self.radar_frame, bg="#111827", height=35)
        legend_frame.pack(fill="x", side="top", padx=0, pady=0)
        tk.Label(legend_frame, text=" 📡  NEON MARITIME COMBAT RANGE SCOPE (EXPANDED PANEL VIEW)", fg="#22D3EE",
                 bg="#111827", font=("Consolas", 10, "bold")).pack(side="left", padx=15, pady=6)
        tk.Label(legend_frame, text="■ Hijacked Alert ", fg="#FF0055", bg="#111827", font=("Consolas", 9, "bold")).pack(
            side="right", padx=15)
        tk.Label(legend_frame, text="■ Genuine Axis ", fg="#00FF66", bg="#111827", font=("Consolas", 9, "bold")).pack(
            side="right", padx=15)

        # Canvas initialization with absolute zero-border profile
        self.canvas = tk.Canvas(self.radar_frame, bg="#010307", bd=0, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=15, pady=15)

    def init_radar_canvas(self):
        self.root.update()
        self.cx = self.canvas.winfo_width() // 2
        self.cy = self.canvas.winfo_height() // 2

        # Enlarge the effective radius coefficient to take over the layout space
        self.radar_radius = min(self.cx, self.cy) - 45

        # Re-scale geospatial markers relative to expanded frame sizes
        self.port_karachi = (self.cx - int(self.radar_radius * 0.75), self.cy + int(self.radar_radius * 0.45))
        self.port_muscat = (self.cx - int(self.radar_radius * 0.15), self.cy - int(self.radar_radius * 0.35))
        self.port_salalah = (self.cx + int(self.radar_radius * 0.70), self.cy + int(self.radar_radius * 0.50))

        self.coast_control_1 = (self.cx - int(self.radar_radius * 0.45), self.cy + int(self.radar_radius * 0.05))
        self.coast_control_2 = (self.cx + int(self.radar_radius * 0.30), self.cy + int(self.radar_radius * 0.15))

        self.draw_static_radar_rings()
        self.sweep_angle = 0
        self.animate_radar_sweep()
        self.animate_pulsing_beacons()

    def draw_static_radar_rings(self):
        self.canvas.delete("static_grid")

        # High-Saturation Depth Gradient Fills
        colors = ["#022B45", "#011F33", "#011421", "#000B14"]
        radii_factors = [1.0, 0.75, 0.5, 0.25]
        for idx, r_fact in enumerate(radii_factors):
            radius = self.radar_radius * r_fact
            self.canvas.create_oval(self.cx - radius, self.cy - radius, self.cx + radius, self.cy + radius,
                                    fill=colors[idx], outline="", tags="static_grid")

        # Neon-Green Range Grid Markers
        for r in [0.25, 0.5, 0.75, 1.0]:
            radius = self.radar_radius * r
            self.canvas.create_oval(self.cx - radius, self.cy - radius, self.cx + radius, self.cy + radius,
                                    outline="#005C2E", width=1, tags="static_grid")

        # Crosshairs
        self.canvas.create_line(self.cx - self.radar_radius, self.cy, self.cx + self.radar_radius, self.cy,
                                fill="#005C2E", tags="static_grid")
        self.canvas.create_line(self.cx, self.cy - self.radar_radius, self.cx, self.cy + self.radar_radius,
                                fill="#005C2E", tags="static_grid")

        # Angular Degree Division Axes
        for angle in [30, 45, 60, 120, 135, 150, 210, 225, 240, 300, 315, 330]:
            rad = math.radians(angle)
            rx = self.cx + self.radar_radius * math.cos(rad)
            ry = self.cy + self.radar_radius * math.sin(rad)
            self.canvas.create_line(self.cx, self.cy, rx, ry, fill="#003319", dash=(2, 4), tags="static_grid")

        # --- COMPASS ROSE HEADINGS OVERLAY ---
        self.canvas.create_text(self.cx, self.cy - self.radar_radius - 14, text="000° N", fill="#00FF66",
                                font=("Consolas", 10, "bold"), tags="static_grid")
        self.canvas.create_text(self.cx + self.radar_radius + 24, self.cy, text="090° E", fill="#00FF66",
                                font=("Consolas", 10, "bold"), tags="static_grid")
        self.canvas.create_text(self.cx, self.cy + self.radar_radius + 14, text="180° S", fill="#00FF66",
                                font=("Consolas", 10, "bold"), tags="static_grid")
        self.canvas.create_text(self.cx - self.radar_radius - 24, self.cy, text="270° W", fill="#00FF66",
                                font=("Consolas", 10, "bold"), tags="static_grid")

        # Port Infrastructure Anchors
        self.canvas.create_text(self.port_karachi[0], self.port_karachi[1] + 20, text="⚓ PORT OF KARACHI\n[PK-ORIGIN]",
                                fill="#22D3EE", font=("Consolas", 9, "bold"), justify="center", tags="static_grid")
        self.canvas.create_text(self.port_muscat[0], self.port_muscat[1] - 20, text="⚓ PORT OF MUSCAT\n[OM-WAYPOINT]",
                                fill="#E0F2FE", font=("Consolas", 9, "bold"), justify="center", tags="static_grid")
        self.canvas.create_text(self.port_salalah[0], self.port_salalah[1] + 20,
                                text="⚓ PORT OF SALALAH\n[OM-TERMINAL]", fill="#F43F5E", font=("Consolas", 9, "bold"),
                                justify="center", tags="static_grid")

        # Secondary Fleet Mapping
        for ship in self.nearby_ships:
            angle_rad = math.radians(ship[0])
            dist = self.radar_radius * ship[1]
            sx = self.cx + dist * math.cos(angle_rad)
            sy = self.cy + dist * math.sin(angle_rad)
            self.canvas.create_polygon(sx, sy - 6, sx - 5, sy + 5, sx + 5, sy + 5, fill="#FACC15", outline="#EAB308",
                                       tags="static_grid")
            self.canvas.create_text(sx + 50, sy + 14, text=f"{ship[2]}", fill="#475569", font=("Consolas", 8, "bold"),
                                    justify="center", tags="static_grid")

    def animate_pulsing_beacons(self):
        self.canvas.delete("pulse_rings")
        self.pulse_r += 1 * self.pulse_dir
        if self.pulse_r > 15 or self.pulse_r < 0:
            self.pulse_dir *= -1

        for port, color in zip([self.port_karachi, self.port_muscat, self.port_salalah],
                               ["#22D3EE", "#A855F7", "#F43F5E"]):
            self.canvas.create_oval(port[0] - self.pulse_r, port[1] - self.pulse_r, port[0] + self.pulse_r,
                                    port[1] + self.pulse_r, outline=color, width=1.5, tags="pulse_rings")

        self.root.after(60, self.animate_pulsing_beacons)

    def animate_radar_sweep(self):
        self.canvas.delete("sweep_line")
        self.sweep_angle = (self.sweep_angle + 4) % 360
        rad = math.radians(self.sweep_angle)
        x_end = self.cx + self.radar_radius * math.cos(rad)
        y_end = self.cy + self.radar_radius * math.sin(rad)

        sweep_color = "#FF0055" if "⚠️" in self.status_indicator.cget("text") else "#00FF66"
        self.canvas.create_line(self.cx, self.cy, x_end, y_end, fill=sweep_color, width=2, tags="sweep_line")
        self.root.after(25, self.animate_radar_sweep)

    def log(self, text):
        self.log_box.insert("end", f">> {text}\n")
        self.log_box.see("end")

    def start_ai_training(self):
        self.btn_train.config(state="disabled", bg="#1E1B4B", fg="#475569")
        self.log("Assembling GRU multi-dimensional deep neural layers...")
        threading.Thread(target=self._worker_train_model, daemon=True).start()

    def _worker_train_model(self):
        try:
            raw_data = generate_synthetic_marinecadastre(num_steps=100)
            spoofed_data = inject_gradual_drift(raw_data, start_idx=65)
            self.dataset_df = calculate_kinematics(spoofed_data)
            self.trained_model = train_gru_model(self.dataset_df, self.features_list, epochs=12)
            self.root.after(0, self._training_complete_ui)
        except Exception as e:
            self.log(f"CRITICAL OVERHEAT: {str(e)}")

    def _training_complete_ui(self):
        self.status_indicator.config(text="AI CORE ONLINE", fg="#22D3EE", bg="#04383F")
        self.btn_simulate.config(state="normal", bg="#00FF66", fg="#04060A")
        self.log("GRU weights assigned. Live tactical feeds validated.")
        messagebox.showinfo("Operational Status", "Dynamic Route Optimization Engines Synchronized!")

    def run_live_simulation(self):
        self.btn_simulate.config(state="disabled", bg="#111827", fg="#475569")
        for item in self.tree.get_children(): self.tree.delete(item)
        self.canvas.delete("ship_track")
        self.draw_static_radar_rings()
        threading.Thread(target=self._worker_simulation, daemon=True).start()

    def get_bezier_point(self, p0, p1, p2, t):
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
        return int(x), int(y)

    def _worker_simulation(self):
        self.trained_model.eval()
        seq_len = 5
        total_steps = len(self.dataset_df) - seq_len
        mid_point = total_steps // 2

        prev_true_x, prev_true_y = None, None
        prev_fake_x, prev_fake_y = None, None
        accumulated_fuel = 0.0

        for i in range(total_steps):
            window_df = self.dataset_df.iloc[i: i + seq_len]
            current_row = window_df.iloc[-1]

            features = window_df[self.features_list].values.astype(np.float32)
            features_tensor = torch.tensor(features).unsqueeze(0)

            with torch.no_grad():
                prediction_prob = self.trained_model(features_tensor).item()

            is_anomaly = prediction_prob > 0.5
            status_text = "DANGER: ATTACK" if is_anomaly else "SECURE"

            if 15 <= i <= 38:
                self.weather_condition = "⚡ STORM TURBULENCE INTERCEPTED"
                wave_height = "6.2m [CRITICAL FORCE]"
                weather_action = "PROPULSION SCALAR ENFORCED"
                weather_color = "#F59E0B"
                weather_strain = 1.7
            else:
                self.weather_condition = "NOMINAL SKIES / CALM CURRENT"
                wave_height = "1.0m [SAFE]"
                weather_action = "PATH CRUISE PROCEEDING"
                weather_color = "#00FF66"
                weather_strain = 1.0

            if i < mid_point:
                self.current_leg = "LEG 1: Karachi ➔ Muscat"
                t = i / mid_point
                true_px, true_py = self.get_bezier_point(self.port_karachi, self.coast_control_1, self.port_muscat, t)
            else:
                if i == mid_point:
                    self.log("Anchored at Port of Muscat.")
                    self.log("Discharging: 2,000 TEU offloaded.")
                    self.total_containers = 2500
                    time.sleep(1.2)
                    self.log("Intake complete: 3,500 TEU secured. Launching toward Salalah...")
                    self.total_containers = 6000

                self.current_leg = "LEG 2: Muscat ➔ Salalah"
                t = (i - mid_point) / (total_steps - mid_point)
                true_px, true_py = self.get_bezier_point(self.port_muscat, self.coast_control_2, self.port_salalah, t)

            true_px += random.randint(-1, 2)
            true_py += random.randint(-1, 1)

            if is_anomaly:
                fake_px = true_px + int((i - 65) * 5)
                fake_py = true_py + int((i - 65) * 3)
            else:
                fake_px, fake_py = true_px, true_py

            self.current_cargo_weight = self.total_containers * 14.2
            total_displacement_mass = self.base_vessel_weight + self.current_cargo_weight
            current_velocity = current_row['SOG'] if not is_anomaly else current_row['V_calc']

            fuel_burn_rate_hr = (0.0000029 * total_displacement_mass) * (current_velocity ** 2.15) * weather_strain
            accumulated_fuel += (fuel_burn_rate_hr / 3600) * 40

            if i < 12:
                zone_str = "SOVEREIGN JURISDICTION (PK)"
                zone_color = "#22D3EE"
            elif i > total_steps - 12:
                zone_str = "EXCLUSIVE ECONOMIC BOUNDARY (OM)"
                zone_color = "#F43F5E"
            else:
                zone_str = "OPEN INTERNATIONAL HIGH SEAS"
                zone_color = "#A855F7"

            timestamp = str(current_row['BaseDateTime']).split('T')[-1]
            v_calc_str = f"{current_row['V_calc']:.2f}"
            sog_str = f"{current_row['SOG']:.1f}"
            delta_str = f"{current_row['Speed_Delta']:.2f}"
            conf_str = f"{prediction_prob * 100:.1f}%"

            self.root.after(0, self._render_ui_tick, timestamp, v_calc_str, sog_str, delta_str, conf_str, status_text,
                            is_anomaly,
                            true_px, true_py, prev_true_x, prev_true_y, fake_px, fake_py, prev_fake_x, prev_fake_y,
                            self.current_leg, self.total_containers, total_displacement_mass,
                            fuel_burn_rate_hr, accumulated_fuel, zone_str, zone_color,
                            self.weather_condition, wave_height, weather_action, weather_color)

            prev_true_x, prev_true_y = true_px, true_py
            prev_fake_x, prev_fake_y = fake_px, fake_py
            time.sleep(0.4)

        self.root.after(0, lambda: self.btn_simulate.config(state="normal", bg="#00FF66", fg="#04060A"))

    def _render_ui_tick(self, timestamp, v_calc, sog, delta, conf, status, is_anomaly,
                        tx, ty, otx, oty, fx, fy, ofx, ofy,
                        leg, containers, total_mass, fuel_rate, total_burn, zone, zone_color,
                        weather, waves, weather_act, weather_col):

        self.lbl_leg.config(text=f"Active Segment:\n👉 {leg}")
        self.lbl_manifest.config(text=f"Containers: {containers} TEU\nMass: {total_mass:,.1f} Metric Tons")
        self.lbl_fuel_rate.config(text=f"Rate: {fuel_rate:.2f} Tons/Hr\nExpended: {total_burn:.2f} Tons")
        self.lbl_zonation.config(text=f"Zone: {zone}", fg=zone_color)
        self.lbl_weather.config(text=f"Matrix: {weather}\nSwells: {waves}")
        self.lbl_weather_action.config(text=weather_act, fg=weather_col)

        row_id = self.tree.insert("", "end", values=(timestamp, v_calc, sog, delta, conf, status))

        # High-thickness trace coordinates
        self.canvas.create_oval(tx - 3, ty - 3, tx + 3, ty + 3, fill="#00FF66", outline="#34D399", tags="ship_track")
        if otx is not None:
            self.canvas.create_line(otx, oty, tx, ty, fill="#00FF66", width=3, tags="ship_track")

        if is_anomaly:
            self.tree.tag_configure("alert", background="#991B1B", foreground="#FEE2E2")
            self.tree.item(row_id, tags=("alert",))
            self.status_indicator.config(text="⚠️ INTRUSION INTERCEPTED", fg="#FF0055", bg="#4C0519")

            self.canvas.create_oval(fx - 3, fy - 3, fx + 3, fy + 3, fill="#FF0055", outline="#FCA5A5",
                                    tags="ship_track")
            if ofx is not None:
                self.canvas.create_line(ofx, ofy, fx, fy, fill="#FF0055", width=2.5, dash=(6, 3), tags="ship_track")

            self.canvas.delete("warning_txt")
            self.canvas.create_text(fx + 65, fy - 14, text="🚨 SPOOF DETECTED", fill="#FF0055",
                                    font=("Consolas", 8, "bold"), tags=("ship_track", "warning_txt"))
        else:
            self.tree.tag_configure("safe", background="#060913", foreground="#E2E8F0")
            self.tree.item(row_id, tags=("safe",))
            if "⚠️" not in self.status_indicator.cget("text"):
                self.status_indicator.config(text="📡 DATA CHANNEL SECURE", fg="#00FF66", bg="#064E3B")

if __name__ == "__main__":
    root = tk.Tk()
    app = MIDSDashboard(root)
    root.mainloop()