import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Label, StringVar, Entry
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class KalmanFilter:
    def __init__(self, F, H, Q, R, P, x):
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P
        self.x = x

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x

    def update(self, z):
        K = np.dot(self.P, self.H.T) / (np.dot(self.H, np.dot(self.P, self.H.T)) + self.R)
        self.x = self.x + K * (z - np.dot(self.H, self.x))
        self.P = (np.eye(len(self.P)) - K * self.H) @ self.P
        return self.x

def update_plot(event=None):
    try:
        Q = np.array([[float(Q_val.get())]])
        R = np.array([[float(R_val.get())]])
        P = np.array([[float(P_val.get())]])
        initial_x = np.array([[float(initial_x_val.get())]])
        offset = float(offset_val.get())
        total_time = float(total_time_val.get())
    except ValueError:
        return
    
    frequency = 1
    amplitude = 5
    sampling_interval = 0.01
    noise_variance = 16
    noise_std_dev = np.sqrt(noise_variance)
    
    kf = KalmanFilter(F=np.array([[1]]), H=np.array([[1]]), Q=Q, R=R, P=P, x=initial_x)

    time_steps = np.arange(0, total_time, sampling_interval)
    true_signal = offset + amplitude * np.sin(2 * np.pi * frequency * time_steps)
    noisy_signal = [val + np.random.normal(0, noise_std_dev) for val in true_signal]

    kalman_estimates = []
    for measurement in noisy_signal:
        kf.predict()
        estimate = kf.update(measurement)
        kalman_estimates.append(estimate[0][0])

    ax.clear()
    ax.plot(time_steps, noisy_signal, label='Noisy Signal', color='orange', alpha=0.6)
    ax.plot(time_steps, true_signal, label='True Signal (Sine Wave)', linestyle='--', color='blue')
    ax.plot(time_steps, kalman_estimates, label='Kalman Filter Estimate', color='green')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Value')
    ax.set_title('Kalman Filter Applied to a Noisy Sinusoidal Wave')
    ax.legend()
    ax.grid()
    canvas.draw()

root = Tk()
root.title("Kalman Filter Parameter Adjustment")
root.geometry("800x600")
root.configure(bg='#f0f0f0')

style = ttk.Style()
style.configure("TLabel", font=('Arial', 10))
style.configure("TEntry", font=('Arial', 10))

frame = ttk.LabelFrame(root, text="Adjust Kalman Filter Parameters", padding=(20, 10))
frame.grid(row=0, column=0, padx=20, pady=10, sticky="ew")

Q_val = StringVar(value="1")
R_val = StringVar(value="10")
P_val = StringVar(value="1")
initial_x_val = StringVar(value="0")
offset_val = StringVar(value="10")
total_time_val = StringVar(value="1")

fields = [
    ("Process Noise Covariance (Q):", Q_val),
    ("Measurement Noise Covariance (R):", R_val),
    ("Initial Estimation Error Covariance (P):", P_val),
    ("Initial State Estimate (x):", initial_x_val),
    ("Signal Offset:", offset_val),
    ("Total Time (s):", total_time_val)
]

for i, (label_text, var) in enumerate(fields):
    label = ttk.Label(frame, text=label_text)
    label.grid(row=i, column=0, padx=5, pady=5, sticky="e")
    entry = ttk.Entry(frame, textvariable=var)
    entry.grid(row=i, column=1, padx=5, pady=5, sticky="w")
    entry.bind("<KeyRelease>", update_plot)

fig, ax = plt.subplots(figsize=(8, 4))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=1, column=0, padx=20, pady=10)

update_plot()

root.mainloop()