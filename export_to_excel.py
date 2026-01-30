import pandas as pd

csv_file = "attendance/attendance.csv"
excel_file = "attendance/attendance.xlsx"

df = pd.read_csv(csv_file)
df.to_excel(excel_file, index=False)

print("Attendance exported to Excel successfully!")
