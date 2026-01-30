import tkinter as tk
import os

def start_system():
    os.system("python main.py")

root = tk.Tk()
root.title("Face Recognition Attendance System")
root.geometry("400x250")
root.resizable(False, False)

tk.Label(
    root,
    text="AI Face Recognition Attendance",
    font=("Arial", 14, "bold")
).pack(pady=20)

tk.Button(
    root,
    text="Start Attendance",
    font=("Arial", 12),
    width=20,
    command=start_system
).pack(pady=10)

tk.Button(
    root,
    text="Exit",
    font=("Arial", 12),
    width=20,
    command=root.destroy
).pack(pady=10)

root.mainloop()
