import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk, messagebox
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas
import os
import datetime
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams.update({"figure.max_open_warning": 0})
sns.set_style("whitegrid")

try:
    data = pd.read_csv("cardio_train.csv", sep=";")
    data = data[(data["ap_hi"] > 40) & (data["ap_hi"] < 250)]
    data = data[(data["ap_lo"] > 40) & (data["ap_lo"] < 200)]
    data = data[(data["height"] > 120) & (data["height"] < 220)]
    data = data[(data["weight"] > 30) & (data["weight"] < 200)]
except FileNotFoundError:
    messagebox.showerror(
        "Hata",
        "cardio_train.csv dosyasi bulunamadi! Lutfen dosyanin uygulamanin calistigi klasorde oldugundan emin olun.",
    )
    exit()

features = [
    "age",
    "gender",
    "height",
    "weight",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
]
X = data[features].copy()
y_sys = data["ap_hi"].copy()
y_dia = data["ap_lo"].copy()

X_train, X_test, y_sys_train, y_sys_test, y_dia_train, y_dia_test = train_test_split(
    X, y_sys, y_dia, test_size=0.2, random_state=42
)


def train_and_evaluate(model, X_tr, y_tr, X_te, y_te, name="Model"):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    mse = mean_squared_error(y_te, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_te, y_pred)
    r2 = r2_score(y_te, y_pred)
    print(f"{name} -> MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.3f}")
    return y_pred, model


print(
    "Model egitimi basliyor — bilgisayariniza bagli olarak bu birkac saniye alabilir..."
)
lr_sys_pred, lr_sys_model = train_and_evaluate(
    LinearRegression(),
    X_train,
    y_sys_train,
    X_test,
    y_sys_test,
    "LinearRegression (Sistolik)",
)
lr_dia_pred, lr_dia_model = train_and_evaluate(
    LinearRegression(),
    X_train,
    y_dia_train,
    X_test,
    y_dia_test,
    "LinearRegression (Diyastolik)",
)

rf_sys_pred, rf_sys_model = train_and_evaluate(
    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    X_train,
    y_sys_train,
    X_test,
    y_sys_test,
    "RandomForest (Sistolik)",
)
rf_dia_pred, rf_dia_model = train_and_evaluate(
    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    X_train,
    y_dia_train,
    X_test,
    y_dia_test,
    "RandomForest (Diyastolik)",
)


def plot_histogram(y_true, y_pred, title="Hata Dagilimi", color="skyblue"):
    plt.figure()
    errors = y_pred - y_true
    plt.hist(errors, bins=40, color=color, edgecolor="black")
    plt.title(title)
    plt.xlabel("Tahmin Hatasi")
    plt.ylabel("Frekans")
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df):
    plt.figure(figsize=(9, 7))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Korelasyon Matrisi")
    plt.tight_layout()
    plt.show()


def plot_age_vs_bp(df, age_col="age"):
    plt.figure()
    vals = df[age_col]
    if vals.mean() > 1000:
        x = vals / 365
        xlabel = "Yas (yil)"
    else:
        x = vals
        xlabel = "Yas"
    plt.scatter(x, df["ap_hi"], alpha=0.35, label="Sistolik", s=10)
    plt.scatter(x, df["ap_lo"], alpha=0.35, label="Diyastolik", s=10)
    plt.xlabel(xlabel)
    plt.ylabel("Kan Basinci (mmHg)")
    plt.title("Yas vs Kan Basinci")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feat_names, title="Ozellik Onemi", color="green"):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(6, 4))
    sns.barplot(x=importances[idx], y=np.array(feat_names)[idx], palette="viridis")
    plt.title(title)
    plt.xlabel("Onem Skoru")
    plt.tight_layout()
    plt.show()


def plot_person_bar(sys_val, dia_val, model_name="Model", person_name="Bilinmiyor"):
    plt.figure()
    sns.barplot(
        x=["Sistolik", "Diyastolik"],
        y=[sys_val, dia_val],
        palette=["skyblue", "salmon"],
    )
    plt.title(f"{person_name} - Ornek Kisi Tahmini ({model_name})")
    plt.ylim(0, max(220, sys_val + 30, dia_val + 30))
    plt.ylabel("Kan Basinci (mmHg)")
    plt.tight_layout()
    plt.show()


def plot_person_on_age(age, sys_val, dia_val, person_name="Bilinmiyor"):
    plt.figure()
    x = age / 365 if age > 1000 else age
    plt.scatter(
        data["age"] / 365 if data["age"].mean() > 1000 else data["age"],
        data["ap_hi"],
        alpha=0.25,
        s=10,
        label="Sistolik (veri)",
    )
    plt.scatter(
        data["age"] / 365 if data["age"].mean() > 1000 else data["age"],
        data["ap_lo"],
        alpha=0.25,
        s=10,
        label="Diyastolik (veri)",
    )
    plt.scatter(
        [x], [sys_val], color="blue", s=120, edgecolor="black", label="Ornek Sistolik"
    )
    plt.scatter(
        [x], [dia_val], color="red", s=120, edgecolor="black", label="Ornek Diyastolik"
    )
    plt.xlabel("Yas (yil)")
    plt.ylabel("Kan Basinci (mmHg)")
    plt.legend()
    plt.title("Ornek Kisi: Yas vs Kan Basinci (Isaretli)")
    plt.tight_layout()
    plt.show()


def plot_all_general():
    plot_histogram(
        y_sys_test,
        rf_sys_pred,
        title="RandomForest (Sistolik) Hata Dagilimi",
        color="lightcoral",
    )
    plot_histogram(
        y_dia_test,
        rf_dia_pred,
        title="RandomForest (Diyastolik) Hata Dagilimi",
        color="lightblue",
    )
    plot_correlation_matrix(data)
    plot_age_vs_bp(data)
    plot_feature_importance(rf_sys_model, features, "Sistolik Ozellik Onemi (RF)")
    plot_feature_importance(rf_dia_model, features, "Diyastolik Ozellik Onemi (RF)")


root = tk.Tk()
root.title("Kan Basinci Tahmin Uygulamasi - Mezuniyet Projesi")
root.state("zoomed")
BG = "#f7fbf7"
FRAME_BG = "#ffffff"
BTN_ACCENT = "#4fb07f"
ERROR_FG = "#b22222"
NORMAL_COLOR = "#0b6b36"
BORDERLINE_COLOR = "#FF8C00"
HIGH_COLOR = "#b22222"
root.configure(bg=BG)

root.grid_rowconfigure(0, weight=0)
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=19)

title = tk.Label(
    root,
    text="KAN BASINCI TAHMIN SISTEMI",
    font=("Helvetica", 20, "bold"),
    fg="#0b6b36",
    bg=BG,
)
title.grid(row=0, column=0, columnspan=2, pady=(12, 6))

main_frame = tk.Frame(root, bg=FRAME_BG, bd=1, relief="flat")
main_frame.grid(row=1, column=0, padx=(14, 7), pady=8, sticky="nsew")

canvas = tk.Canvas(main_frame, bg=FRAME_BG, highlightthickness=0)
scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas, bg=FRAME_BG)
scrollable_frame.bind(
    "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

model_select_frame = tk.Frame(scrollable_frame, bg=FRAME_BG)
model_select_frame.pack(padx=12, pady=(12, 6), anchor="w")
model_label = tk.Label(
    model_select_frame, text="Model Secimi:", bg=FRAME_BG, font=("Helvetica", 11)
)
model_label.pack(side="left", padx=(0, 10))
model_choice = ttk.Combobox(
    model_select_frame,
    values=["Random Forest", "Linear Regression"],
    state="readonly",
    width=20,
)
model_choice.current(0)
model_choice.pack(side="left")

form_frame = tk.Frame(scrollable_frame, bg=FRAME_BG)
form_frame.pack(padx=12, pady=6, anchor="w")

labels_and_values = {
    "Yas (yil)": (tk.Scale, {"from_": 20, "to": 80, "orient": "horizontal"}),
    "Cinsiyet": (
        ttk.Combobox,
        {"values": ["Erkek (1)", "Kadin (2)"], "state": "readonly"},
    ),
    "Boy (cm)": (tk.Scale, {"from_": 140, "to": 200, "orient": "horizontal"}),
    "Kilo (kg)": (tk.Scale, {"from_": 40, "to": 150, "orient": "horizontal"}),
    "Kolesterol": (
        ttk.Combobox,
        {"values": ["Normal (1)", "Sinirda (2)", "Yuksek (3)"], "state": "readonly"},
    ),
    "Glukoz": (
        ttk.Combobox,
        {"values": ["Normal (1)", "Sinirda (2)", "Yuksek (3)"], "state": "readonly"},
    ),
    "Sigara": (
        ttk.Combobox,
        {"values": ["Hayir (0)", "Evet (1)"], "state": "readonly"},
    ),
    "Alkol": (ttk.Combobox, {"values": ["Hayir (0)", "Evet (1)"], "state": "readonly"}),
    "Aktif": (ttk.Combobox, {"values": ["Hayir (0)", "Evet (1)"], "state": "readonly"}),
}

entries = {}
row_idx = 0
for lab, (widget_type, params) in labels_and_values.items():
    lbl = tk.Label(form_frame, text=lab + " :", anchor="e", width=22, bg=FRAME_BG)
    lbl.grid(row=row_idx, column=0, padx=6, pady=3, sticky="e")

    if widget_type == tk.Scale:
        ent = widget_type(form_frame, **params)
        ent.set(70 if "Kilo" in lab else (170 if "Boy" in lab else 40))
    else:
        ent = widget_type(form_frame, **params)
        ent.current(0)

    ent.grid(row=row_idx, column=1, padx=12, pady=3, sticky="w")
    entries[lab] = ent
    row_idx += 1

name_lbl = tk.Label(form_frame, text="Isim :", anchor="e", width=22, bg=FRAME_BG)
name_lbl.grid(row=row_idx, column=0, padx=6, pady=3, sticky="e")
name_ent = tk.Entry(form_frame, width=20, justify="center")
name_ent.grid(row=row_idx, column=1, padx=12, pady=3, sticky="w")
entries["Isim"] = name_ent
row_idx += 1

surname_lbl = tk.Label(form_frame, text="Soyisim :", anchor="e", width=22, bg=FRAME_BG)
surname_lbl.grid(row=row_idx, column=0, padx=6, pady=3, sticky="e")
surname_ent = tk.Entry(form_frame, width=20, justify="center")
surname_ent.grid(row=row_idx, column=1, padx=12, pady=3, sticky="w")
entries["Soyisim"] = surname_ent

result_var = tk.StringVar()
result_label = tk.Label(
    scrollable_frame,
    textvariable=result_var,
    font=("Helvetica", 13, "bold"),
    fg=ERROR_FG,
    bg=FRAME_BG,
    justify="center",
)
result_label.pack(pady=(6, 8))

btn_frame = tk.Frame(scrollable_frame, bg=FRAME_BG)
btn_frame.pack(pady=(6, 14), anchor="w")


def setup_database():
    """Veritabanini olusturur ve tabloyu ayarlar."""
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT,
            age REAL,
            gender INTEGER,
            height REAL,
            weight REAL,
            cholesterol INTEGER,
            gluc INTEGER,
            smoke INTEGER,
            alco INTEGER,
            active INTEGER,
            model_name TEXT,
            sys_pred REAL,
            dia_pred REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.commit()
    conn.close()


def save_to_database(
    full_name,
    age,
    gender,
    height,
    weight,
    cholesterol,
    gluc,
    smoke,
    alco,
    active,
    model_name,
    sys_pred,
    dia_pred,
):
    """Verileri veritabanina kaydeder."""
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO predictions (
            full_name, age, gender, height, weight, cholesterol, gluc, smoke, alco, active, model_name, sys_pred, dia_pred
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            full_name,
            age,
            gender,
            height,
            weight,
            cholesterol,
            gluc,
            smoke,
            alco,
            active,
            model_name,
            sys_pred,
            dia_pred,
        ),
    )
    conn.commit()
    conn.close()


def load_history_from_db():
    """Veritabanindan gecmis verileri Treeview'a yukler."""
    for item in tree.get_children():
        tree.delete(item)
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = c.fetchall()
    for row in rows:
        tree.insert(
            "",
            "end",
            values=(
                row[1],  # full_name
                row[2],  # age
                row[3],  # gender
                row[4],  # height
                row[5],  # weight
                row[6],  # cholesterol
                row[7],  # gluc
                row[8],  # smoke
                row[9],  # alco
                row[10],  # active
                row[11],  # model_name
                round(row[12], 1),  # sys_pred
                round(row[13], 1),  # dia_pred
            ),
        )
    conn.close()


def save_to_pdf(
    full_name,
    age,
    gender,
    height,
    weight,
    cholesterol,
    gluc,
    smoke,
    alco,
    active,
    model_name,
    sys_val,
    dia_val,
):
    pdf_file = f"Rapor_{full_name.replace(' ','_')}.pdf"
    c = Canvas(pdf_file, pagesize=A4)
    # Raporlab'de Turkce karakter kullanmak icin font ayari yapilmasi gerekir,
    # ancak ASCII uyumu istendigi icin sadece mevcut fontu koruyup
    # metinleri ASCII uyumlu yapiyoruz.
    c.setFont("Helvetica-Bold", 18)
    c.drawString(120, 800, "Kan Basinci Tahmin Raporu")
    c.setFont("Helvetica", 10)
    c.drawRightString(
        550, 785, f"Tarih: {datetime.datetime.now().strftime('%d-%m-%Y')}"
    )

    c.setFont("Helvetica", 12)
    c.line(100, 765, 500, 765)
    c.drawString(100, 750, "Kisisel Bilgiler")
    c.setFont("Helvetica", 11)

    info_y = 730
    c.drawString(120, info_y, f"Isim Soyisim: {full_name}")
    c.drawString(120, info_y - 20, f"Yas: {age} yil")
    c.drawString(120, info_y - 40, f"Cinsiyet: {'Erkek' if gender == 1 else 'Kadin'}")
    c.drawString(120, info_y - 60, f"Boy: {height} cm")
    c.drawString(120, info_y - 80, f"Kilo: {weight} kg")

    c.line(100, info_y - 95, 500, info_y - 95)
    c.drawString(100, info_y - 110, "Saglik Verileri")
    c.setFont("Helvetica", 11)

    c.drawString(120, info_y - 130, f"Kolesterol: {cholesterol}")
    c.drawString(120, info_y - 150, f"Glukoz: {gluc}")
    c.drawString(120, info_y - 170, f"Sigara: {'Evet' if smoke == 1 else 'Hayir'}")
    c.drawString(120, info_y - 190, f"Alkol: {'Evet' if alco == 1 else 'Hayir'}")
    c.drawString(
        120, info_y - 210, f"Fiziksel Aktiflik: {'Evet' if active == 1 else 'Hayir'}"
    )

    c.line(100, info_y - 225, 500, info_y - 225)
    c.drawString(100, info_y - 240, "Model Tahminleri")
    c.setFont("Helvetica-Bold", 12)

    c.drawString(120, info_y - 260, f"Secilen Model: {model_name}")
    c.drawString(
        120, info_y - 280, f"Tahmin Edilen Sistolik Kan Basinci: {sys_val:.1f} mmHg"
    )
    c.drawString(
        120, info_y - 300, f"Tahmin Edilen Diyastolik Kan Basinci: {dia_val:.1f} mmHg"
    )

    c.save()
    print(f"PDF kaydedildi: {pdf_file}")


def make_prediction(show_general_graphs=False, compare=False):
    try:
        age = entries["Yas (yil)"].get()
        gender_str = entries["Cinsiyet"].get()
        gender = int(gender_str.split(" ")[1].replace("(", "").replace(")", ""))
        height = entries["Boy (cm)"].get()
        weight = entries["Kilo (kg)"].get()
        cholesterol_str = entries["Kolesterol"].get()
        cholesterol = int(
            cholesterol_str.split(" ")[1].replace("(", "").replace(")", "")
        )
        gluc_str = entries["Glukoz"].get()
        gluc = int(gluc_str.split(" ")[1].replace("(", "").replace(")", ""))
        smoke_str = entries["Sigara"].get()
        smoke = int(smoke_str.split(" ")[1].replace("(", "").replace(")", ""))
        alco_str = entries["Alkol"].get()
        alco = int(alco_str.split(" ")[1].replace("(", "").replace(")", ""))
        active_str = entries["Aktif"].get()
        active = int(active_str.split(" ")[1].replace("(", "").replace(")", ""))

    except Exception as e:
        result_var.set(f"Girdi hatasi: {e}")
        return

    sel = model_choice.get()
    person_df = pd.DataFrame(
        [[age, gender, height, weight, cholesterol, gluc, smoke, alco, active]],
        columns=features,
    )

    if sel == "Linear Regression":
        sys_pred = lr_sys_model.predict(person_df)[0]
        dia_pred = lr_dia_model.predict(person_df)[0]
    else:
        sys_pred = rf_sys_model.predict(person_df)[0]
        dia_pred = rf_dia_model.predict(person_df)[0]

    full_name = f"{entries['Isim'].get()} {entries['Soyisim'].get()}" or "Bilinmiyor"

    if sys_pred >= 140 or dia_pred >= 90:
        result_label.config(fg=HIGH_COLOR)
    elif (120 <= sys_pred < 140) or (80 <= dia_pred < 90):
        result_label.config(fg=BORDERLINE_COLOR)
    else:
        result_label.config(fg=NORMAL_COLOR)

    result_var.set(
        f"{sel} ile Tahmin -> Sistolik: {sys_pred:.1f} mmHg  Diyastolik: {dia_pred:.1f} mmHg"
    )

    save_to_database(
        full_name,
        age,
        gender,
        height,
        weight,
        cholesterol,
        gluc,
        smoke,
        alco,
        active,
        sel,
        sys_pred,
        dia_pred,
    )

    add_log_entry(
        full_name,
        age,
        gender,
        height,
        weight,
        cholesterol,
        gluc,
        smoke,
        alco,
        active,
        sel,
        sys_pred,
        dia_pred,
    )

    save_to_pdf(
        full_name,
        age,
        gender,
        height,
        weight,
        cholesterol,
        gluc,
        smoke,
        alco,
        active,
        sel,
        sys_pred,
        dia_pred,
    )

    plot_person_bar(sys_pred, dia_pred, sel, person_name=full_name)
    plot_person_on_age(age, sys_pred, dia_pred, person_name=full_name)

    if show_general_graphs:
        plot_all_general()

    if compare:
        lr_sys_v = lr_sys_model.predict(person_df)[0]
        lr_dia_v = lr_dia_model.predict(person_df)[0]
        plt.figure(figsize=(6, 4))
        x = np.arange(2)
        width = 0.35
        plt.bar(
            x - width / 2,
            [lr_sys_v, lr_dia_v],
            width,
            label="LinearRegression",
            color="#a9a9a9",
        )
        plt.bar(x + width / 2, [sys_pred, dia_pred], width, label=sel, color="#2e8b57")
        plt.xticks(x, ["Sistolik", "Diyastolik"])
        plt.ylabel("mmHg")
        plt.title(f"Model Karsilastirma: LinearRegression vs {sel}")
        plt.legend()
        plt.tight_layout()
        plt.show()


def clear_history():
    """Treeview ve veritabanindaki tum verileri temizler."""
    if messagebox.askyesno(
        "Gecmisi Temizle",
        "Tum gecmisi silmek istediginizden emin misiniz? Bu islem geri alinamaz.",
    ):
        conn = sqlite3.connect("predictions.db")
        c = conn.cursor()
        c.execute("DELETE FROM predictions")
        conn.commit()
        conn.close()
        for item in tree.get_children():
            tree.delete(item)


predict_button = tk.Button(
    btn_frame,
    text="Tahmin Et (Secili Model)",
    bg="#2e8b57",
    fg="white",
    font=("Helvetica", 11, "bold"),
    command=lambda: make_prediction(show_general_graphs=False, compare=False),
)
predict_button.grid(row=0, column=0, padx=8, pady=6)

predict_plus_graphs_button = tk.Button(
    btn_frame,
    text="Tahmin + Tum Grafikleri Goster",
    bg="#4682b4",
    fg="white",
    font=("Helvetica", 11, "bold"),
    command=lambda: make_prediction(show_general_graphs=True, compare=False),
)
predict_plus_graphs_button.grid(row=1, column=0, padx=8, pady=6)

compare_button = tk.Button(
    btn_frame,
    text="Model Karsilastir (LR vs Secili)",
    bg="#8a2be2",
    fg="white",
    font=("Helvetica", 11, "bold"),
    command=lambda: make_prediction(show_general_graphs=False, compare=True),
)
compare_button.grid(row=2, column=0, padx=8, pady=6)

clear_button = tk.Button(
    btn_frame,
    text="Gecmisi Temizle",
    bg="#ff6347",
    fg="white",
    font=("Helvetica", 11, "bold"),
    command=clear_history,
)
clear_button.grid(row=3, column=0, padx=8, pady=6)


log_frame = tk.Frame(root, bg=BG)
log_frame.grid(row=1, column=1, padx=(7, 14), pady=8, sticky="nsew")
log_frame.grid_rowconfigure(1, weight=1)
log_frame.grid_columnconfigure(0, weight=1)

log_label = tk.Label(
    log_frame, text="Tahmin Gecmisi", font=("Helvetica", 14, "bold"), bg=BG
)
log_label.grid(row=0, column=0, sticky="w", pady=(0, 6), padx=6)

columns = [
    "Isim Soyisim",
    "Yas",
    "Cinsiyet",
    "Boy",
    "Kilo",
    "Kolesterol",
    "Glukoz",
    "Sigara",
    "Alkol",
    "Aktif",
    "Model",
    "Sistolik",
    "Diyastolik",
]
tree = ttk.Treeview(log_frame, columns=columns, show="headings")
for col in columns:
    tree.heading(col, text=col)
    tree.column(col, width=80, anchor="center")
tree.grid(row=1, column=0, sticky="nsew")

log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=tree.yview)
tree.configure(yscrollcommand=log_scroll.set)
log_scroll.grid(row=1, column=1, sticky="ns")


def add_log_entry(
    full_name,
    age,
    gender,
    height,
    weight,
    cholesterol,
    gluc,
    smoke,
    alco,
    active,
    model_name,
    sys_val,
    dia_val,
):
    tree.insert(
        "",
        "end",
        values=[
            full_name,
            age,
            gender,
            height,
            weight,
            cholesterol,
            gluc,
            smoke,
            alco,
            active,
            model_name,
            round(sys_val, 1),
            round(dia_val, 1),
        ],
    )


setup_database()
load_history_from_db()
root.mainloop()
