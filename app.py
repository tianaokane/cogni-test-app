import streamlit as st
import pandas as pd
import random
import time
from datetime import datetime
import os
import matplotlib.pyplot as plt
import joblib


# Password for admin view
# ADMIN_PASSWORD = "brainwave"

# ------------------------ SETUP ------------------------
st.set_page_config(page_title="Stroop Test", page_icon="🧠", layout="wide")

# ------------------------ STYLING ------------------------
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-size: 20px !important;
        }
        h1 {
            font-size: 48px !important;
            font-weight: bold;
        }
        .stRadio > div {
            font-size: 20px !important;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 3rem;
            padding-right: 3rem;
            max-width: 1200px;
            margin: auto;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------ INITIAL STATE ------------------------
if "page" not in st.session_state:
    st.session_state.page = "intro"
if "results" not in st.session_state:
    st.session_state.results = []
if "trial" not in st.session_state:
    st.session_state.trial = 0
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "word" not in st.session_state:
    st.session_state.word = None
if "font_color" not in st.session_state:
    st.session_state.font_color = None
if "submit_clicked" not in st.session_state:
    st.session_state.submit_clicked = False

# ------------------------ COLOURS ------------------------
colors = ["Red", "Blue", "Green", "Yellow"]
color_map = {"Red": "red", "Blue": "blue", "Green": "green", "Yellow": "yellow"}
NUM_TRIALS = 5

# ------------------------ FUNCTION ------------------------
def setup_trial():
    st.session_state.word = random.choice(colors)
    st.session_state.font_color = random.choice(colors)
    st.session_state.start_time = time.time()
    
def reset():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.page = "intro"
    st.experimental_rerun()


# ------------------------ INTRO ------------------------
if st.session_state.page == "intro":
    st.header("🧠 Welcome to the Stroop Test")
    st.markdown("""
    This is a short interactive cognitive task.

    🟡 You'll be shown a **color word** (e.g., "Red")  
    🔵 The word will be in a different **font color**  
    🟢 Your task is to choose the **font color**, *not the word*  
    ⏱ Your **accuracy and speed** will be measured

    ---

    Please answer the following before starting:
    """)
    st.session_state.user_meta = {
        "tiredness": st.slider("How tired are you? (0 = energetic, 10 = exhausted)", 0, 10, 5),
        "caffeine": st.radio("Have you had caffeine today?", ["Yes", "No"]),
        "focus": st.slider("How focused do you feel?", 0, 10, 5),
        "adhd": st.radio("Do you have ADHD or an attention disorder?", ["Yes", "No", "Prefer not to say"])
    }
    if st.button("Start Test"):
        setup_trial()
        st.session_state.page = "test"
        st.rerun()

# ------------------------ TEST ------------------------
elif st.session_state.page == "test":
    st.title("🧠 Stroop Test - Cognitive Task")

    # 🌈 Add pastel gradient progress bar here
    progress = (st.session_state.trial / NUM_TRIALS) * 100
    st.markdown(f"""
        <div style='background: linear-gradient(to right, #ffd1dc, #cce7ff);
                    height: 10px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    overflow: hidden;'>
            <div style='background-color: #91c788;
                        width: {progress}%;
                        height: 100%;
                        border-radius: 10px;
                        transition: width 0.3s ease-in-out;'>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<h1 style='color:{color_map[st.session_state.font_color]}'>{st.session_state.word}</h1>", unsafe_allow_html=True)
    choice = st.radio("What color is the TEXT?", colors, key=f"choice_{st.session_state.trial}", index=None)
    st.session_state.current_choice = choice

    if st.button("Submit"):
        st.session_state.submit_clicked = True
        st.rerun()

    if st.session_state.submit_clicked:
        if st.session_state.current_choice is None:
            st.warning("Please select an answer before submitting.")
        else:
            rt = round(time.time() - st.session_state.start_time, 3)
            correct = (st.session_state.current_choice == st.session_state.font_color)#
            congruent = (st.session_state.word == st.session_state.font_color)

            st.session_state.results.append({
                "trial": st.session_state.trial + 1,
                "word": st.session_state.word,
                "font_color": st.session_state.font_color,
                "response": st.session_state.current_choice,
                "correct": correct,
                "reaction_time": rt,
                "congruent": congruent,
                **st.session_state.user_meta
            })

            st.session_state.trial += 1
            st.session_state.submit_clicked = False

            if st.session_state.trial >= NUM_TRIALS:
                st.session_state.page = "explanation"
            else:
                setup_trial()
            st.rerun()


# ------------------------ EXPLANATION ------------------------
elif st.session_state.page == "explanation":
    st.title("🧠 Why was this test challenging?")
    st.markdown("""
    The Stroop Test is designed to create a conflict in your brain: the word says one thing, but the font color says another.

    🔁 Your brain processes **words automatically**.  
    🎯 But in this task, you're asked to **focus on the font color**, not the word.

    This creates **cognitive interference**, especially when the word and color don’t match (incongruent trials). It’s a classic measure of attention, reaction control, and processing speed.
    
    ---
    
    Let's see how you did.
    """)
    if st.button("Show My Results"):
        st.session_state.page = "done"
        st.rerun()

# ------------------------ RESULTS ------------------------
elif st.session_state.page == "done":
    df = pd.DataFrame(st.session_state.results)
    st.success("Test complete! ✅")

    score = df["correct"].sum()
    total = len(df)
    avg_rt = df["reaction_time"].mean()

    st.markdown(f"### 🎯 Score: **{score} / {total}**")
    st.markdown(f"### ⏱ Avg Reaction Time: **{avg_rt:.3f} seconds**")

    # Load the pre-trained model
    model_path = "model/stroop_rf_model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)

        # Predict on current session
        latest_session = df.copy()
        latest_summary = {
            "reaction_time": latest_session["reaction_time"].mean(),
            "correct": latest_session["correct"].mean(),
            "congruent": latest_session["congruent"].astype(int).mean(),
            "tiredness": latest_session["tiredness"].iloc[0],
            "focus": latest_session["focus"].iloc[0],
            "caffeine": 1 if latest_session["caffeine"].iloc[0] == "Yes" else 0,
            "adhd": {"Yes": 1, "No": 0, "Prefer not to say": -1}[latest_session["adhd"].iloc[0]],
        }
        latest_df = pd.DataFrame([latest_summary])

        # Make prediction
        prediction = model.predict(latest_df)[0]
        st.markdown(f"### 🤖 Our model predicts you were: **{prediction.upper()}** during this session.")
    else:
        st.warning("ML model not found. Please train and save it first.")

    # Load the master dataset
    master_path = "data/results.csv"
    if not os.path.exists(master_path):
        st.error("No master results.csv file found to train a model.")
        st.stop()

    master_df = pd.read_csv(master_path)

    # Preprocess
    master_df["caffeine"] = master_df["caffeine"].map({"Yes": 1, "No": 0})
    master_df["adhd"] = master_df["adhd"].map({"Yes": 1, "No": 0, "Prefer not to say": -1})
    master_df["congruent"] = master_df["congruent"].astype(int)
    master_df["correct"] = master_df["correct"].astype(int)

    # Aggregate features by session
    user_summary = master_df.groupby("session_id").agg({
        "reaction_time": "mean",
        "correct": "mean",
        "congruent": "mean",
        "tiredness": "first",
        "focus": "first",
        "caffeine": "first",
        "adhd": "first"
    }).reset_index()

    # Label training data
    user_summary["attention_state"] = user_summary.apply(
        lambda row: "distracted" if (row["correct"] < 0.7 or row["reaction_time"] > 2.5 or row["focus"] < 4) else "focused", axis=1
    )

    # Train model
    X = user_summary.drop(columns=["session_id", "attention_state"])
    y = user_summary["attention_state"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Predict on current session
    latest_session = df.copy()
    latest_summary = {
        "reaction_time": latest_session["reaction_time"].mean(),
        "correct": latest_session["correct"].mean(),
        "congruent": latest_session["congruent"].astype(int).mean(),
        "tiredness": latest_session["tiredness"].iloc[0],
        "focus": latest_session["focus"].iloc[0],
        "caffeine": 1 if latest_session["caffeine"].iloc[0] == "Yes" else 0,
        "adhd": {"Yes": 1, "No": 0, "Prefer not to say": -1}[latest_session["adhd"].iloc[0]],
    }
    latest_df = pd.DataFrame([latest_summary])

    # Make prediction
    prediction = model.predict(latest_df)[0]

    # Show result to user
    st.markdown(f"### 🤖 Our model predicts you were: **{prediction.upper()}** during this session.")


    if st.button("🔁 Retake Test"):
        reset()

    st.markdown("---")
    st.markdown("#### Want to see group results or advanced analysis?")
    if "request_admin" not in st.session_state:
        st.session_state.request_admin = False

    if st.button("🔐 Admin View"):
        st.session_state.request_admin = True
        st.rerun()

    if st.session_state.request_admin and st.session_state.page == "done":
        st.markdown("---")
        st.subheader("🔐 Enter Admin Password to View Full Statistics")
        password_input = st.text_input("Password", type="password")
        if password_input == "brainwave":
            st.session_state.page = "admin"
            st.session_state.request_admin = False
            st.rerun()
        elif password_input:
            st.error("Incorrect password.")

# ------------------------ ADMIN ------------------------
elif st.session_state.page == "admin":
    df = pd.DataFrame(st.session_state.results)

    st.title("📊 Full Statistics")
    st.dataframe(df)

    st.markdown("### ✅ Overall Accuracy")
    st.write(f"{df['correct'].mean() * 100:.1f}%")

    st.markdown("### 🎯 Accuracy by Trial Type")
    acc_by_type = df.groupby("congruent")["correct"].mean().reset_index()
    acc_by_type["congruent"] = acc_by_type["congruent"].map({True: "Congruent", False: "Incongruent"})
    st.dataframe(acc_by_type.rename(columns={"congruent": "Trial Type", "correct": "Accuracy"}))

    st.markdown("### ⏱ Reaction Time by Trial Type")
    rt_by_type = df.groupby("congruent")["reaction_time"].mean().reset_index()
    rt_by_type["congruent"] = rt_by_type["congruent"].map({True: "Congruent", False: "Incongruent"})
    st.dataframe(rt_by_type.rename(columns={"congruent": "Trial Type", "reaction_time": "Avg Reaction Time (s)"}))

    st.markdown("### 📈 Reaction Time Over Trials")
    fig, ax = plt.subplots()
    ax.plot(df["trial"], df["reaction_time"], marker='o', linestyle='-', color='purple')
    ax.set_xlabel("Trial")
    ax.set_ylabel("Reaction Time (s)")
    ax.set_title("Reaction Time Per Trial")
    st.pyplot(fig)

    st.markdown("### 🧠 Focus and Attention Meta-data")
    focus_summary = df[["tiredness", "focus"]].describe().loc[["mean", "min", "max"]]
    st.dataframe(focus_summary)

    st.markdown("### 👥 Group Accuracy by ADHD Status")
    grouped_acc = df.groupby("adhd")["correct"].mean().reset_index()
    st.dataframe(grouped_acc.rename(columns={"correct": "Accuracy"}))

    st.markdown("### ☕ Group Accuracy by Caffeine")
    caffeine_acc = df.groupby("caffeine")["correct"].mean().reset_index()
    st.dataframe(caffeine_acc.rename(columns={"correct": "Accuracy"}))

    st.markdown("### 🔢 Avg Reaction Time by Focus Level")
    focus_bins = pd.cut(df["focus"], bins=[0,3,6,10], labels=["Low", "Medium", "High"])
    focus_rt = df.groupby(focus_bins)["reaction_time"].mean().reset_index()
    focus_rt.columns = ["Focus Level", "Avg Reaction Time (s)"]
    st.dataframe(focus_rt)


# Save to CSV
df = pd.DataFrame(st.session_state.results)

os.makedirs("data", exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
session_id = f"session_{timestamp}"

# Add tracking fields
df["timestamp"] = timestamp
df["session_id"] = session_id

# Save individual session file
filename = f"data/results_{timestamp}.csv"
df.to_csv(filename, index=False)
st.success(f"Results saved to `{filename}`")

# Append to master file
master_path = "data/results.csv"
if os.path.exists(master_path):
    df.to_csv(master_path, mode='a', header=False, index=False)
else:
    df.to_csv(master_path, index=False)


