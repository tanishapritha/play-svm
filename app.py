import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Linear SVM App", layout="wide")

st.title("Linear SVM: Scratch vs scikit-learn")

uploaded_file = st.file_uploader("Upload CSV file (binary classification)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    all_columns = df.columns.tolist()
    feature_cols = st.multiselect("Select feature columns:", all_columns, default=all_columns[:2])
    label_col = st.selectbox("Select label column:", all_columns, index=len(all_columns)-1)
    X = df[feature_cols].values
    y = df[label_col].values
    uniques = np.unique(y)
    if len(uniques) != 2:
        st.error("Please upload a binary classification CSV (2 classes).")
        st.stop()
    y_scratch = np.where(y == uniques[0], -1, 1)
else:
    st.info("Using default noisy linear dataset")
    X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1,
                               class_sep=0.7, flip_y=0.2, random_state=42)
    y_scratch = np.where(y == 0, -1, 1)
    df = pd.DataFrame(X, columns=[f"Feature {i+1}" for i in range(X.shape[1])])
    df['Label'] = y

# ---------------- Dataset Preview ----------------
st.subheader("Dataset Preview (Toy data by default)")
st.dataframe(df.head())

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_scratch, test_size=0.2, random_state=42)

# Scratch SVM
def train_scratch_svm(X, y, lr=0.005, lambda_param=0.01, n_iters=5000):
    w = np.zeros(X.shape[1])
    b = 0
    for _ in range(n_iters):
        idxs = np.random.permutation(len(X))
        for i in idxs:
            x_i, y_i = X[i], y[i]
            if y_i * (np.dot(x_i, w) + b) >= 1:
                dw = 2 * lambda_param * w
                db = 0
            else:
                dw = 2 * lambda_param * w - x_i * y_i
                db = -y_i
            w -= lr * dw
            b -= lr * db
    return w, b

w_scratch, b_scratch = train_scratch_svm(X_train, y_train)
y_pred_scratch = np.sign(X_test @ w_scratch + b_scratch)
acc_scratch = accuracy_score(y_test, y_pred_scratch)
report_scratch = classification_report(y_test, y_pred_scratch, output_dict=True)

# Function to plot decision boundary
def plot_boundary(X, y, w=None, b=None, model=None, title=""):
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    grid = np.c_[xx.ravel(), yy.ravel()]
    if model is not None:
        Z = model.predict(grid).reshape(xx.shape)
    else:
        Z = np.sign(grid @ w + b).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.contourf(xx, yy, Z, alpha=0.2, cmap='bwr')
    ax.scatter(X[:,0], X[:,1], c=y, cmap='bwr', s=50)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(title)
    return fig

tab1, tab2 = st.tabs(["Comparison", "Play with Parameters"])

# ---------------- Comparison Tab ----------------
with tab1:
    st.subheader("Scratch vs scikit-learn SVM")

    # Train default scikit-learn SVM
    svm = SVC(kernel='linear', C=1.0)
    svm.fit(X_train, y_train)
    y_pred_sklearn = svm.predict(X_test)
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
    report_sklearn = classification_report(y_test, y_pred_sklearn, output_dict=True)

    col1, col2 = st.columns(2)
    # Scratch SVM
    with col1:
        st.markdown("### Scratch SVM")
        with st.expander("View Scratch SVM Code"):
            st.code('''w = np.zeros(X_train.shape[1])
b = 0
lr = 0.005
lambda_param = 0.01
n_iters = 5000

for _ in range(n_iters):
    idxs = np.random.permutation(len(X_train))
    for i in idxs:
        x_i, y_i = X_train[i], y_train[i]
        if y_i * (np.dot(x_i, w) + b) >= 1:
            dw = 2 * lambda_param * w
            db = 0
        else:
            dw = 2 * lambda_param * w - x_i * y_i
            db = -y_i
        w -= lr * dw
        b -= lr * db''', language="python")
        fig1 = plot_boundary(X_test, y_test, w=w_scratch, b=b_scratch, title="Scratch SVM Boundary")
        st.pyplot(fig1)
        st.markdown("**Accuracy**")
        st.table(pd.DataFrame({"Accuracy":[f"{acc_scratch*100:.2f}%"]}))
        st.markdown("**Classification Report**")
        st.dataframe(pd.DataFrame(report_scratch).transpose().style.format("{:.2f}"))

    # scikit-learn SVM
    with col2:
        st.markdown("### scikit-learn SVM")
        with st.expander("View scikit-learn SVM Code"):
            st.code('''from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)''', language="python")
        fig2 = plot_boundary(X_test, y_test, model=svm, title="scikit-learn SVM Boundary")
        st.pyplot(fig2)
        st.markdown("**Accuracy**")
        st.table(pd.DataFrame({"Accuracy":[f"{acc_sklearn*100:.2f}%"]}))
        st.markdown("**Classification Report**")
        st.dataframe(pd.DataFrame(report_sklearn).transpose().style.format("{:.2f}"))

# ---------------- Play with Parameters Tab ----------------
with tab2:
    st.subheader("Adjust scikit-learn SVM Parameters")

    C_val = st.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.01)
    max_iter_val = st.slider("Max Iterations", 100, 5000, 1000, 50)

    # Use the same noisy dataset (or uploaded CSV)
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Standardize features for better boundary responsiveness
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_scaled)
    X_test_scaled = scaler.transform(X_test_scaled)

    # Retrain SVM with new parameters
    svm_param = SVC(kernel='linear', C=C_val, max_iter=max_iter_val)
    svm_param.fit(X_train_scaled, y_train)
    y_pred_param = svm_param.predict(X_test_scaled)
    
    acc_param = accuracy_score(y_test, y_pred_param)
    report_param = classification_report(y_test, y_pred_param, output_dict=True)

    # Side-by-side plot + metrics
    col1p, col2p = st.columns(2)

    with col1p:
        fig_param = plot_boundary(X_test_scaled, y_test, model=svm_param,
                                  title=f"SVM Boundary (C={C_val})")
        st.pyplot(fig_param)

    with col2p:
        st.markdown("**Metrics**")
        st.table(pd.DataFrame({"Accuracy":[f"{acc_param*100:.2f}%"]}))
        st.markdown("**Classification Report**")
        st.dataframe(pd.DataFrame(report_param).transpose().style.format("{:.2f}"))
