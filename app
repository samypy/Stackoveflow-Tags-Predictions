import tkinter
from tkinter import *
#from tkinter import ttk
from tkinter import messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from openpyxl import Workbook
import sqlite3
import socket

hostname = socket.gethostname()
print(hostname)
from datetime import date
#from tkinter import ttk
# Function to create the activity log table in the SQL database
def create_activity_log_table():
    conn = sqlite3.connect('activity_log.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS activity_log
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  date TEXT,
                  activity_type TEXT,
                  requestor_soe TEXT,
                  requestor_business TEXT,
                  report_type TEXT,
                  time_spent INTEGER)''')
    conn.commit()
    conn.close()


def clear():
    query = "SELECT id, date, activity_type, requestor_soe, requestor_business, report_type, time_spent FROM activity_log"
    conn = sqlite3.connect('activity_log.db')
    c = conn.cursor()
    c.execute(query)
    rows= c.fetchall()
    print(rows)
    update(rows)
    conn.commit()

# Function to insert a new activity log into the SQL database
def insert_activity_log(date, activity_type, requestor_soe, requestor_business, report_type, time_spent):
    conn = sqlite3.connect('activity_log.db')
    c = conn.cursor()
    c.execute("INSERT INTO activity_log (date, activity_type, requestor_soe, requestor_business, report_type, time_spent) VALUES (?, ?, ?, ?, ?, ?)", (date, activity_type, requestor_soe, requestor_business, report_type, time_spent))
    conn.commit()
    conn.close()

def get_activity_logs(date=None, activity_type=None, requestor_soe=None, requestor_business=None, report_type=None, time_spent=None):
    conn = sqlite3.connect('activity_log.db')
    c = conn.cursor()
    query = "SELECT * FROM activity_log"
    if date or activity_type or requestor_soe or requestor_business or report_type or time_spent:
        query += " WHERE"
        if date:
            query += f" date = '{date}' AND"
        if activity_type:
            query += f" activity_type = '{activity_type}' AND"
        if requestor_soe:
            query += f" requestor_soe = '{requestor_soe}' AND"
        if requestor_business:
            query += f" requestor_business = '{requestor_business}' AND"
        if report_type:
            query += f" report_type = '{report_type}' AND"
        if time_spent:
            query += f" time_spent = {time_spent} AND"
        query = query[:-4]
    c.execute(query)
    activity_logs = c.fetchall()
    print(activity_logs)
    conn.close()
    return activity_logs

def update_activity_logs():
    date = t2.get()
    activity_type = t3.get()
    requestor_soe = t4.get()
    requestor_business = t5.get()
    report_type = t6.get()
    time_spent = t7.get()
    id = t1.get()
    conn = sqlite3.connect('activity_log.db')
    c = conn.cursor()
    if messagebox.askyesno("Confrim update?", "Are you sure you want to update this customer"):
        query = "UPDATE activity_log SET date = ?, activity_type = ?, requestor_soe = ?, requestor_business = ?, report_type = ?, time_spent = ? WHERE id=?"
        c.execute(query, (date, activity_type, requestor_soe, requestor_business, report_type, time_spent, id))
        conn.commit()
        date_entry.entry.delete(0, ttk.END)
        activity_combobox.delete(0, ttk.END)
        requestor_soe_entry.delete(0, ttk.END)
        requestor_business_entry.delete(0, ttk.END)
        activity_combobox.delete(0, ttk.END)
        duration_entry.delete(0, ttk.END)
    clear()

def update(rows):
    activity_table.delete(*activity_table.get_children())
    for i in rows:
        activity_table.insert('', 'end', values=i)

def getrow(event):
    rowdid = activity_table.identify_row(event.y)
    item = activity_table.item(activity_table.focus())
    print(item['values'][0])
    t1.set(item['values'][0])
    t2.set(item['values'][1])
    t3.set(item['values'][2])
    t4.set(item['values'][3])
    t5.set(item['values'][4])
    t6.set(item['values'][5])
    t7.set(item['values'][6])

def delete_log():
    id = t1.get()
    print(id)
    if messagebox.askyesno("Confrim Delete?", "Are you sure you want to delete this customer"):
        query = "DELETE FROM activity_log WHERE id = "+id
        conn = sqlite3.connect('activity_log.db')
        c = conn.cursor()
        c.execute(query)
        clear()
    else:
        return True

def download_excel():
    # Create a new Excel workbook
    workbook = Workbook()

    # Get the active worksheet
    worksheet = workbook.active

    # Write the column headers to the worksheet
    headings = []
    for col in activity_table["columns"]:
        headings.append(activity_table.heading(col)["text"])
    worksheet.append(headings)

    activity_logs = get_activity_logs()
    for activity_log in activity_logs:
        worksheet.append(activity_log)
    # Write the data to the worksheet
    # for item in activity_table.get_children():
    #     values = []
    #     for col in activity_table["columns"]:
    #         values.append(activity_table.item(item)["values"][int(col[1])-1])
    #     worksheet.append(values)

    # Save the workbook to a file
    try:
        workbook.save("treeview_data.xlsx")
        messagebox.showinfo("Excel Download", "Data has been exported to 'Activity_data.xlsx' successfully!")
    except Exception as e:
        messagebox.showerror("Excel Download", f"An error occurred while exporting data: {e}")

window = ttk.Window(themename="flatly")
window.title("Daily Activity Tracker")

t1 = StringVar()
t2 = StringVar()
t3 = StringVar()
t4 = StringVar()
t5 = StringVar()
t6 = StringVar()
t7 = StringVar()
frame = ttk.Frame(window)
frame.grid()

# Saving User Info
bold_font = ('Arial', 12, 'bold')
user_info_frame = ttk.LabelFrame(frame, text="Activity Details", bootstyle="primary")
user_info_frame.grid(row=0, column=0, sticky="news", padx=20, pady=10)


date_label = ttk.Label(user_info_frame, text="Activity Date", bootstyle="info")
date_label.grid(row=0, column=0,pady=10,padx=10)
date_entry = ttk.DateEntry(user_info_frame, bootstyle="success")
date_entry.grid(row=1, column=0,pady=10,padx=10)

activity_label = ttk.Label(user_info_frame, text="Activity Type", bootstyle="info")
activity_combobox = ttk.Combobox(user_info_frame, values=["Regular Report", "Adhoc Reports"], bootstyle="success", textvariable=t3)
activity_label.grid(row=0, column=1,pady=10,padx=10)
activity_combobox.grid(row=1, column=1,pady=10,padx=10)


# first_name_entry = tkinter.Entry(user_info_frame)
# last_name_entry = tkinter.Entry(user_info_frame)
# first_name_entry.grid(row=1, column=0)
# last_name_entry.grid(row=1, column=1)

report_label = ttk.Label(user_info_frame, text="Report Type", bootstyle="info")
report_combobox = ttk.Combobox(user_info_frame, values=["Readership", "Publication"], bootstyle="success", textvariable=t6)
report_label.grid(row=0, column=3,pady=10,padx=10)
report_combobox.grid(row=1, column=3,pady=10,padx=10)

# for widget in user_info_frame.winfo_children():
#     widget.grid_configure(padx=10, pady=5)

# nationality_label = tkinter.Label(user_info_frame, text="Nationality")
# nationality_combobox = ttk.Combobox(user_info_frame,
#                                     values=["Africa", "Antarctica", "Asia", "Europe", "North America", "Oceania",
#                                             "South America"])
# nationality_label.grid(row=2, column=1)
# nationality_combobox.grid(row=3, column=1)
#
# for widget in user_info_frame.winfo_children():
#     widget.grid_configure(padx=10, pady=5)
#
# Saving Course Info
Requestor_frame = ttk.LabelFrame(frame, text="Requestor Details")
Requestor_frame.grid(row=1, column=0, sticky="news", padx=20, pady=10)

# Requestor_label = ttk.Label(Requestor_frame, text="Requestor Details")

requestor_soe_label = ttk.Label(Requestor_frame, text="Requestor SOE")
requestor_soe_entry = ttk.Entry(Requestor_frame, textvariable=t4)
requestor_soe_label.grid(row=0, column=0, pady=10,padx=10)
requestor_soe_entry.grid(row=1, column=0, pady=10,padx=10)

# requestor_name_label = ttk.Label(Requestor_frame, text="Requestor Name")
# requestor_name_entry = ttk.Entry(Requestor_frame, textvariable=t7)
# requestor_name_label.grid(row=0, column=1, pady=10,padx=10)
# requestor_name_entry.grid(row=1, column=1, pady=10,padx=10)

requestor_business_label = ttk.Label(Requestor_frame, text="Requestor Business")
requestor_business_entry = ttk.Entry(Requestor_frame, textvariable=t5)
requestor_business_label.grid(row=0, column=1, pady=10,padx=10)
requestor_business_entry.grid(row=1, column=1, pady=10,padx=10)

duration_label = ttk.Label(Requestor_frame, text="Time Spent")
duration_entry = ttk.Entry(Requestor_frame, textvariable=t7)
duration_label.grid(row=0, column=2, pady=10,padx=10)
duration_entry.grid(row=1, column=2, pady=10,padx=10)


def submit_activity_log():
    # Get the values of the fields
    date = date_entry.entry.get()
    activity_type = activity_combobox.get()
    report_type = report_combobox.get()
    requestor_soe = requestor_soe_entry.get()
    requestor_business = requestor_business_entry.get()
    time_spent = duration_entry.get()

    # Insert the activity log into the SQL database
    insert_activity_log(date, activity_type, requestor_soe, requestor_business, report_type, time_spent)

    # Clear the fields
    activity_combobox.set('Regular report')
    #report_combobox.set('Readership report')
    report_combobox.delete(0, ttk.END)
    requestor_soe_entry.delete(0, ttk.END)
    requestor_business_entry.delete(0, ttk.END)
    duration_entry.delete(0, ttk.END)
    activity_logs = get_activity_logs()
    for activity_log in activity_logs:
        activity_table.insert("", "end", values=activity_log)
# Button
button = tkinter.Button(Requestor_frame, text="Submit", command=submit_activity_log)
button.grid(row=1, column=4, sticky="news", padx=20, pady=10)



activity_table = ttk.Treeview(frame, columns=("ID","Date", "Activity Type", "Requestor SOE", "Requestor Business", "Report Type", "Time_Spent"), bootstyle="info")
activity_table.heading("ID", text="ID", anchor="w")
activity_table.heading("Date", text="Date", anchor="w")
activity_table.heading("Activity Type", text="Activity Type", anchor="w")
activity_table.heading("Requestor SOE", text="Requestor SOE", anchor="w")
activity_table.heading("Requestor Business", text="Requestor Business", anchor="w")
activity_table.heading("Report Type", text="Report Type", anchor="w")
activity_table.heading("Time_Spent", text="Time Spent (in minutes)", anchor="w")
#activity_table.pack()
activity_table.grid(sticky="news", padx=20, pady=10)
activity_table.bind('<Double 1>', getrow)

query = "SELECT id, date, activity_type, requestor_soe, requestor_business, report_type, time_spent FROM activity_log"
conn = sqlite3.connect('activity_log.db')
c = conn.cursor()
c.execute(query)
conn.commit()
rows = c.fetchall()
update(rows)


# Add a button to download the data as an Excel file
download_button = ttk.Button(frame, text="Download Excel", command=download_excel)
download_button.grid()


# create buttons frame
buttons_frame = ttk.Frame(frame)
buttons_frame.grid(row=1, column=0, pady=10)

# create edit and delete buttons
edit_button = ttk.Button(buttons_frame, text="Update", command=update_activity_logs)
delete_button = ttk.Button(buttons_frame, text="Delete", command=delete_log)

# add edit and delete buttons to the frame
edit_button.grid(row=0, column=0, padx=5)
delete_button.grid(row=0, column=1, padx=5)




# function to be called when delete button is clicked
def delete_activity_log():
    # get the currently selected item in the treeview
    selected_item = activity_table.selection()
    if not selected_item:
        messagebox.showerror("Error", "No activity log selected.")
        return

    # ask for confirmation before deleting the item
    if messagebox.askyesno("Confirm Deletion", "Are you sure you want to delete this activity log?"):
        # delete the selected item from the treeview
        activity_table.delete(selected_item)


# function to be called when edit button is clicked
def edit_activity_log():
    # get the currently selected item in the treeview
    selected_item = activity_table.selection(activity_table.focus())
    if not selected_item:
        messagebox.showerror("Error", "No activity log selected.")
        return

    # get the values of the selected item
    values = activity_table.item(selected_item, "values")

    # fill the entry widgets with the selected values
    date_entry.entry.delete(0, tkinter.END)
    date_entry.entry.insert(0, values[0])
    activity_combobox.set(values[1])
    report_combobox.set(values[2])

    # delete the selected item from the treeview
    activity_table.delete(selected_item)

# # create edit and delete buttons
# edit_button = ttk.Button(buttons_frame, text="Update", command=edit_activity_log)
# delete_button = ttk.Button(buttons_frame, text="Delete", command=delete_activity_log)




# Fetch all the activity logs from the SQL database and add them to the table
activity_logs = get_activity_logs()
for activity_log in activity_logs:
    activity_table.insert("", "end", values=activity_log)

window.mainloop()

if __name__ == '__main__':
    submit_activity_log()
