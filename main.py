import sys
import os
from tkinter import *
from ttkbootstrap import Style
from PIL import Image, ImageTk
from openpyxl.utils.dataframe import dataframe_to_rows
from ttkbootstrap.dialogs.dialogs import Querybox
from ttkbootstrap.toast import ToastNotification
import ttkbootstrap as ttkb
import tkinter.filedialog as filedialog
import numpy as np
import pandas as pd
import cv2
import openpyxl
import datetime
import time
import segment as sg

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

root = Tk()
root.title('UltraScan')
root.geometry("1600x900")
root.resizable(False, False)
style = Style(theme="litera")

icon_path = resource_path("images/main_icon.png")
if os.path.exists(icon_path):
    main_icon = ImageTk.PhotoImage(file=icon_path)
    root.iconphoto(False, main_icon)
else:
    print(f"Error: The file {icon_path} does not exist.")

headerFont = "Helvetica"
bodyFont = "Verdana"

btn_style = ttkb.Style().configure(style="my.TButton", font=(bodyFont, 30))
btn_style1 = ttkb.Style().configure(style="heading.TButton", font=(headerFont, 30))

canvas = Canvas(root, width=1600, height=900, highlightthickness=0)
canvas.pack(fill="both", expand=True)

def startScreen():
    print("Starting screen")
    canvas.delete("all")
    bg_image = Image.open(resource_path("images/PAGE1_BG.png"))
    bg_image = bg_image.resize((1600, 900))
    bg_image_tk = ImageTk.PhotoImage(bg_image)
    canvas.create_image(800, 450, image=bg_image_tk, anchor="center")
    canvas.image = bg_image_tk
    
    # get started button
    start_img = Image.open(resource_path("images/PAGE1BUTTON_GETSTARTED.png"))
    start_img = start_img.resize((415, 69))
    start_img_tk = ImageTk.PhotoImage(start_img)
    
    start_btn = canvas.create_image(1035, 655, image=start_img_tk, anchor="center")
    canvas.tag_bind(start_btn, "<Button-1>", lambda event: homeScreen())  # bind click event to go homescreen
    print("Start screen setup complete")

def homeScreen():
    global bg_image_tk
    print("Home screen")
    canvas.delete("all")
    bg_image = Image.open(resource_path("images/SEGMENT PAGE_EMPTY_NEW.png"))
    bg_image = bg_image.resize((1600, 900))
    bg_image_tk = ImageTk.PhotoImage(bg_image)
    canvas.create_image(800, 450, image=bg_image_tk, anchor="center")
    canvas.image = bg_image_tk

    select_img = Image.open(resource_path("images/PAGE2BUTTON_SELECTIMAGE.png"))
    select_img = select_img.resize((415, 69))
    select_img_tk = ImageTk.PhotoImage(select_img)

    selectImg_btn = canvas.create_image(615, 750, image=select_img_tk, anchor="center")
    canvas.tag_bind(selectImg_btn, "<Button-1>", lambda event: selectImg())
    canvas.select_image = select_img_tk  # reference to avoid garbage collection

    question_img = Image.open(resource_path("images/PAGEBUTTON_QUESTION.png"))
    question_img = question_img.resize((40, 38))
    question_img_tk = ImageTk.PhotoImage(question_img)
    questionImg_btn = canvas.create_image(1400, 100, image=question_img_tk, anchor="center")
    canvas.tag_bind(questionImg_btn, "<Button-1>", lambda event: userManualScreen())
    canvas.question_image = question_img_tk 

    print("Home screen setup complete")

def userManualScreen():
    print("User Manual screen")
    canvas.delete("all")
    bg_image = Image.open(resource_path("images/MANUAL PAGE_EMPTY.png"))
    bg_image = bg_image.resize((1600, 900))
    bg_image_tk = ImageTk.PhotoImage(bg_image)
    canvas.create_image(800, 450, image=bg_image_tk, anchor="center")
    canvas.image = bg_image_tk
    
    proceed_img = Image.open(resource_path("images/UNDERSTAND_BUTTON.png"))
    proceed_img = proceed_img.resize((415, 69))
    proceed_img_tk = ImageTk.PhotoImage(proceed_img)
    proceed_btn = canvas.create_image(1035, 750, image=proceed_img_tk, anchor="center")
    canvas.tag_bind(proceed_btn, "<Button-1>", lambda event: homeScreen())
    canvas.proceed_image = proceed_img_tk

def segmentScreen(image):
    print("Segment screen")
    bpd, ofd, hc, gestAge, angle = None, None, None, None, None
    
    # Segmented image
    start_time = time.process_time()
    result = sg.segment(image)
    end_time = time.process_time()
    elapsed_time = end_time - start_time
    print(f"Process took {elapsed_time:.4f} seconds to complete")
    if result is None:
        print("An error occurred during segmentation")
        notifToast("Segmentation Failed",
        "Please choose a suitable image.",
        "danger")
        return
    else:
        segmented_image, semi_axes_a, semi_axes_b, angle = result

    # Background image
    canvas.delete("all")
    bg_image = Image.open(resource_path("images/PAGE_3.png"))
    bg_image = bg_image.resize((1600, 900))
    bg_image_tk = ImageTk.PhotoImage(bg_image)
    canvas.create_image(800, 450, image=bg_image_tk, anchor="center")
    canvas.image = bg_image_tk

    def calculateParams(a, b, angle):
        try:
            nonlocal bpd, ofd, hc, gestAge
            pixelsize = float(my_entry.get())
            if pixelsize <= 0:
                raise ValueError
            bpd, ofd, hc, gestAge = getParameters(pixelsize, a, b)
            print(f"Parameters calculated: bpd={bpd}, ofd={ofd}, hc={hc}, gestAge={gestAge}, angle={angle}")
            canvas.itemconfig(angle_item, text=f'{np.round(angle, 2)} degrees')
            bpd = np.round(bpd, 2)
            ofd = np.round(ofd, 2)
            hc = np.round(hc, 2)

            canvas.itemconfig(bpd_item, text=f'{bpd} mm')
            canvas.itemconfig(ofc_item, text=f'{ofd} mm')
            canvas.itemconfig(hc_item, text=f'{hc} cm')
            canvas.itemconfig(gestAge_item, text=f'{gestAge} weeks')

            save_btn = canvas.create_image(525, 738, image=save_img_tk, anchor="center")
            canvas.tag_bind(save_btn, "<Button-1>", lambda event: saveInfo(angle, bpd, ofd, hc, gestAge, save_btn))
        except ValueError:
            notifToast("Pixel Size Warning",
                    "Please input an integer greater than 0.",
                    "danger")
            print("ValueError in calculateParams")

    # Load button images
    calculate_img = Image.open(resource_path("images/PAGE3BUTTON_CALCULATE.png"))
    calculate_img = calculate_img.resize((219, 69))
    calculate_img_tk = ImageTk.PhotoImage(calculate_img)

    save_img = Image.open(resource_path("images/PAGE3BUTTON_SAVE.png"))
    save_img = save_img.resize((219, 69))
    save_img_tk = ImageTk.PhotoImage(save_img)

    back_img = Image.open(resource_path("images/PAGE3BUTTON_ENTER.png"))
    back_img = back_img.resize((40, 38))
    back_img_tk = ImageTk.PhotoImage(back_img)

    # Create buttons with images
    calculate_btn = canvas.create_image(230, 738, image=calculate_img_tk, anchor="center")
    canvas.tag_bind(calculate_btn, "<Button-1>", lambda event: calculateParams(semi_axes_a, semi_axes_b, angle))

    back_btn = canvas.create_image(125, 100, image=back_img_tk, anchor="center")
    canvas.tag_bind(back_btn, "<Button-1>", lambda event: homeScreen())

    canvas.calculate_image = calculate_img_tk
    canvas.back_image = back_img_tk

    print(f"Segmented image: semi_axes_a={semi_axes_a}, semi_axes_b={semi_axes_b}, angle={angle}")
    segmented_image = Image.fromarray(segmented_image)  # Convert image to PIL format
    segmented_image = segmented_image.resize((640, 400))  # Resize image
    segmented_image_tk = ImageTk.PhotoImage(segmented_image)  # Convert image to PhotoImage
    canvas.create_image(1200, 450, image=segmented_image_tk, anchor="center")  # Adjusted position to the right
    canvas.segmented_image = segmented_image_tk  # Keep a reference to avoid garbage collection

    notifToast("Segmentation Successful!",
            "The image has been successfully segmented.",
            "success")

    my_entry = ttkb.Entry(canvas,
                          bootstyle="info",
                          font=(bodyFont, 18),
                          width=14)

    canvas.create_window(440, 300, anchor="nw", window=my_entry)

    canvas.create_text(385, 300, text="Pixel Size:", anchor="ne", font=(bodyFont, 20, "bold"), fill="black")
    canvas.create_text(385, 350, text="Angle:", anchor="ne", font=(bodyFont, 20, "bold"), fill="black")
    canvas.create_text(385, 400, text="BPD:", anchor="ne", font=(bodyFont, 20, "bold"), fill="black")
    canvas.create_text(385, 450, text="OFD:", anchor="ne", font=(bodyFont, 20, "bold"), fill="black")
    canvas.create_text(385, 500, text="HC:", anchor="ne", font=(bodyFont, 20, "bold"), fill="black")
    canvas.create_text(97, 550, text="Gestational Age:", anchor="nw", font=(bodyFont, 20, "bold"), fill="black")

    angle_item = canvas.create_text(440, 350, anchor="nw", text="", font=(bodyFont, 20), fill="black")
    bpd_item = canvas.create_text(440, 400, anchor="nw", text="", font=(bodyFont, 20), fill="black")
    ofc_item = canvas.create_text(440, 450, anchor="nw", text="", font=(bodyFont, 20), fill="black")
    hc_item = canvas.create_text(440, 500, anchor="nw", text="", font=(bodyFont, 20), fill="black")
    gestAge_item = canvas.create_text(440, 550, text="", anchor="nw", font=(bodyFont, 20), fill="black")

    print("Segment screen setup complete")

def getParameters(pixelSize, a, b):
    print("Calculating parameters")
    semi_axes_a_mm = a * pixelSize / 2
    semi_axes_b_mm = b * pixelSize / 2

    bpd = semi_axes_a_mm * 2
    ofd = semi_axes_b_mm * 2

    hc = 1.62 * (bpd + ofd)
    hc = np.round(hc / 10, 3)

    conditions = [
        (hc < 8.00),
        (hc >= 8.00) & (hc <= 9.00),  # week 13
        (hc > 9.01) & (hc <= 10.49),  # week 14
        (hc > 10.50) & (hc <= 12.49),  # week 15
        (hc > 12.50) & (hc <= 13.49),  # week 16
        (hc > 13.50) & (hc <= 14.99),  # week 17
        (hc > 15.00) & (hc <= 16.49),  # week 18
        (hc > 16.50) & (hc <= 17.49),  # week 19
        (hc > 17.50) & (hc <= 18.99),  # week 20
        (hc > 19.00) & (hc <= 19.99),  # week 21
        (hc > 20.00) & (hc <= 20.99),  # week 22
        (hc > 21.00) & (hc <= 22.49),  # week 23
        (hc > 22.50) & (hc <= 22.99),  # week 24
        (hc >= 23.00) & (hc <= 23.99),  # week 25
        (hc > 24.00) & (hc <= 24.79),  # week 26
        (hc > 24.80) & (hc <= 25.60),  # week 27
        (hc > 25.61) & (hc <= 26.75),  # week 28
        (hc > 26.76) & (hc <= 27.75),  # week 29
        (hc > 27.76) & (hc <= 28.85),  # week 30
        (hc > 28.86) & (hc <= 29.60),  # week 31
        (hc > 29.61) & (hc <= 30.40),  # week 32
        (hc > 30.41) & (hc <= 31.20),  # week 33
        (hc > 31.21) & (hc <= 31.80),  # week 34
        (hc > 31.81) & (hc <= 32.50),  # week 35
        (hc > 32.51) & (hc <= 33.00),  # week 36
        (hc > 33.01) & (hc <= 33.70),  # week 37
        (hc > 33.71) & (hc <= 34.20),  # week 38
        (hc > 34.21) & (hc <= 35.00),  # week 39
        (hc > 35.00) & (hc <= 36.00),  # week 40
        (hc > 36)
    ]

    values = ['Fetus is less than 8 Menstrual Weeks', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22',
              '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
              '40', 'Abnormal']

    return bpd, ofd, hc, np.select(conditions, values)

def selectImg():
    file_path = filedialog.askopenfilename(title="Select Image",
                                           filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        print("Selected file:", file_path)  # to be edited
        imgNp = cv2.imread(file_path, cv2.IMREAD_COLOR)

        img = Image.open(file_path)
        img = img.resize((640, 360))
        img_tk = ImageTk.PhotoImage(img)

        canvas.create_image(800, 470, image=img_tk, anchor="center")
        canvas.image = img_tk

        canvas.create_text(800, 675, text="Image successfully chosen!", anchor="center", font=(bodyFont, 15), fill="black")

        segment_img = Image.open(resource_path("images/PAGE2BUTTON_SEGMENT.png"))
        segment_img = segment_img.resize((415, 69))
        segment_img_tk = ImageTk.PhotoImage(segment_img)

        segment_btn = canvas.create_image(1095, 750, image=segment_img_tk, anchor="center")
        canvas.tag_bind(segment_btn, "<Button-1>", lambda event: segmentScreen(imgNp))
        canvas.segment_image = segment_img_tk

def saveInfo(angle, bpd, ofd, hc, gestAge, button):
    name = Querybox.get_string(prompt='Enter name to be saved:', title='Save Info')
    if name is None:
        return
    elif name == '':
        notifToast("Save Unsuccessful",
        "Name should not be blank.",
        "danger")
        return

    date = datetime.date.today().strftime("%m/%d/%Y")  # Convert date to string

    # Create a pandas DataFrame from the data
    data = {'Name': [name], 'Angle': [angle], 'BPD': [bpd], 'OFD': [ofd], 'HC': [hc], 'Age (wk)': [np.array2string(gestAge)], 'Date Added': [date]}
    df = pd.DataFrame(data)

    # Define the file path and name
    file_path = "Fetal Head.xlsx"

    # Check if the file exists
    if os.path.exists(file_path):
        # Load the existing workbook
        wb = openpyxl.load_workbook(file_path)
        ws = wb.active
        # Append the data to the worksheet
        for r in dataframe_to_rows(df, index=False, header=False):
            ws.append(r)
    else:
        # Create a new workbook
        wb = openpyxl.Workbook()
        ws = wb.active
        # Write the header row
        for r in dataframe_to_rows(df, index=False, header=True):
            ws.append(r)

    # Save the workbook
    wb.save(file_path)
    notifToast("Save Successful",
               "Successfully saved at Fetal Head.xlsx",
               "success")
    canvas.itemconfig(button, state=HIDDEN)


def notifToast(title, msg, style):
    ToastNotification(title=title,
                                message=msg,
                                duration=3000,
                                bootstyle=style).show_toast()

startScreen()
root.mainloop()
