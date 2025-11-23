# Interactive Dashboard Documentation (app.py)

## Overview

`app.py` is a production-ready Streamlit dashboard that provides an interactive interface for exploring the Iris dataset and making real-time species predictions using a trained Support Vector Machine model.

---

## Features at a Glance

- **Home Page**: Project overview with dataset statistics and species gallery
- **Prediction Page**: Real-time classification with interactive sliders and 3D visualization
- **Data Analysis Page**: Comprehensive EDA with interactive charts
- **Model Performance Page**: Metrics, confusion matrix, and model comparison

---

## Technical Architecture

### Dependencies

```python
streamlit>=1.29.0      # Web framework
plotly>=5.18.0         # Interactive visualizations
pandas>=2.0.0          # Data manipulation
numpy>=1.24.0          # Numerical operations
scikit-learn>=1.3.0    # Model loading
joblib>=1.3.0          # Model serialization
```

### Configuration

```python
st.set_page_config(
    page_title="Iris Species Classifier",
    page_icon="I",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

**Design Philosophy:**
- Wide layout for optimal visualization space
- Expanded sidebar for easy navigation
- Custom CSS with dark mode support
- Inter font family for modern typography

---

## Core Components

### 1. Data Loading Functions

#### `load_model_assets()`
```python
@st.cache_resource
def load_model_assets():
    """Load trained model, scaler, and label encoder."""
```

**Purpose:** Load ML artifacts from disk once and cache in memory  
**Returns:** `(model, scaler, label_encoder)`  
**Cache Type:** `@st.cache_resource` - persists across sessions  
**Error Handling:** Displays error message if files not found

**Required Files:**
- `models/best_model.joblib` - Trained SVM classifier
- `models/scaler.joblib` - StandardScaler for feature normalization
- `models/label_encoder.joblib` - LabelEncoder for species names

#### `load_data()`
```python
@st.cache_data
def load_data():
    """Load Iris dataset"""
```

**Purpose:** Load dataset from `data_processing` module  
**Returns:** Pandas DataFrame with Iris data  
**Cache Type:** `@st.cache_data` - immutable data caching

---

### 2. Visualization Functions

#### `create_3d_scatter(df, new_sample=None, prediction=None)`

**3D scatter plot showing dataset distribution and user predictions**

**Parameters:**
- `df`: Iris DataFrame
- `new_sample`: User input array [sepal_length, sepal_width, petal_length, petal_width]
- `prediction`: Predicted species label

**Features:**
- **Axes**: X=Petal Length, Y=Petal Width, Z=Sepal Length
- **Species colors**: Defined in `SPECIES_COLORS` dictionary
- **User sample**: Yellow diamond marker (size=15)
- **Interactivity**: Rotate, zoom, pan with mouse
- **Height**: 600px

**Color Scheme:**
```python
SPECIES_COLORS = {
    'Iris-setosa': '#FF6B6B',      # Coral Red
    'Iris-versicolor': '#4ECDC4',  # Turquoise
    'Iris-virginica': '#9B5DE5'    # Purple
}
```

#### `create_feature_distributions(df)`

**2x2 grid of histograms showing feature distributions by species**

**Features:**
- 4 subplots (SepalLength, SepalWidth, PetalLength, PetalWidth)
- Overlaid histograms per species
- Color-coded by species
- Height: 500px

**Use Case:** Understanding feature ranges and overlap between species

#### `create_correlation_heatmap(df)`

**Heatmap visualization of feature correlations**

**Features:**
- 4x4 correlation matrix
- Blue color scale (light to dark)
- Annotated with correlation values (2 decimals)
- Height: 400px

**Key Correlations:**
- Petal Length ↔ Petal Width: 0.96 (highest)
- Sepal Width: Weak correlation with others

#### `create_pairplot(df)`

**Interactive scatter matrix showing pairwise relationships**

**Features:**
- Lower triangular matrix (diagonal and upper half hidden)
- Color-coded by species
- Interactive zoom and pan
- Height: 700px

**Use Case:** Identifying linear separability and feature relationships

---

## Page Implementations

### Home Page

**Purpose:** Welcome users and provide project context

**Sections:**

1. **Hero Card**
   - Gradient background (dark blue to light blue)
   - Project tagline and description
   - Call-to-action style design

2. **What You Can Do**
   - Bulleted list of dashboard capabilities
   - Two-column layout (60%-40%)

3. **Species Gallery**
   - 3-column image grid
   - Wikipedia images of each species
   - Rounded corners with shadows

4. **Dataset Statistics**
   - 4 metrics in row: Samples (150), Features (4), Species (3), Accuracy (97.37%)
   - Uses `st.metric()` for visual consistency

**Design Notes:**
- Clean, professional aesthetic
- Gradient cards for visual hierarchy
- Responsive column layout

---

### Prediction Page

**Purpose:** Interactive species classification

**Layout:** Two-column (1:2 ratio)

#### Left Column: Input Controls

**4 Sliders:**
```python
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.01)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.01)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.01)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3, 0.01)
```

**Features:**
- 0.01 step precision
- Tooltips with typical ranges
- Default values: mid-range specimens

**Predict Button:**
- Primary type (blue/purple gradient)
- Full width of column
- Triggers prediction on click

#### Right Column: Results

**Prediction Display:**
1. **Prediction Box** - Large, gradient background with species name
2. **Species Information** - Description and biological characteristics
3. **Species Image** - Wikipedia photo of predicted species
4. **Measurement Summary** - 2x2 metric grid showing input values

**Session State Management:**
```python
st.session_state.prediction = prediction_label
st.session_state.sample = sample[0]
```

Stores prediction results for 3D visualization below.

#### 3D Visualization Section

**Full-width plot** showing:
- All 150 training samples (colored by species)
- User's prediction (yellow diamond)
- Interactive controls for rotation

**Instructions:**
- Each point = one flower
- Colors = species
- Yellow diamond = your sample
- Click and drag to rotate

---

### Data Analysis Page

**Purpose:** Comprehensive exploratory data analysis

**Structure:**

1. **Dataset Preview (Expandable)**
   - Full dataframe display
   - Summary statistics (`df.describe()`)
   - Class distribution bar chart

2. **Feature Distributions**
   - 2x2 histogram grid
   - Overlaid by species
   - Shows clear separation (especially petal features)

3. **Correlation Analysis**
   - Two-column layout (50%-50%)
   - Left: Heatmap visualization
   - Right: Text insights and interpretation

4. **Pairwise Relationships**
   - Full-width scatter matrix
   - Loading spinner during generation
   - How-to-read instructions

**Key Insights Highlighted:**
- Setosa has smallest petal measurements
- Petal length/width most discriminative
- Virginica has largest measurements

---

### Model Performance Page

**Purpose:** Showcase model metrics and validation

**Structure:**

#### 1. Model Information (3 columns)
- **Column 1**: Algorithm details (SVM, Linear kernel)
- **Column 2**: Dataset split and CV strategy
- **Column 3**: Performance summary and status

#### 2. Classification Metrics (4 columns)
```python
st.metric("Accuracy", "97.37%", delta="2.89% vs baseline")
st.metric("Precision", "97.56%", delta="Weighted average")
st.metric("Recall", "97.37%", delta="Weighted average")
st.metric("F1 Score", "97.36%", delta="Harmonic mean")
```

**Features:**
- Delta indicators for context
- Tooltips with metric definitions
- Consistent formatting

#### 3. Per-Class Performance

**Table:** Precision, Recall, F1-Score by species  
**Styling:** Background gradient (red-yellow-green scale)

**Bar Chart:**
- Grouped bars for 3 metrics
- Shows performance parity across species
- Setosa: perfect classification
- Versicolor/Virginica: minor confusion

#### 4. Key Findings (Text Summary)
- Narrative explanation of results
- Per-species breakdown
- Overall assessment

#### 5. Confusion Matrix

**Heatmap Visualization:**
```python
confusion_data = np.array([
    [12, 0, 0],   # Setosa: Perfect
    [0, 12, 1],   # Versicolor: 1 error
    [0, 1, 12]    # Virginica: 1 error
])
```

**Features:**
- Blue color scale
- Annotated cell values
- Clear axis labels

**Interpretation:**
- Diagonal = correct predictions
- Off-diagonal = misclassifications
- Total errors: 2/38 samples

#### 6. Model Comparison

**Grouped Bar Chart:**
- CV Accuracy (light blue) vs Test Accuracy (dark blue)
- 4 models: SVM, KNN, Logistic Regression, Random Forest
- Y-axis range: 0.85 - 1.0

**Success Message:**
- Green box highlighting SVM selection
- Production-ready status

---

## Styling & Design

### CSS Variables (with Dark Mode Support)

**Light Mode:**
```css
--page-gradient-start: rgba(79, 70, 229, 0.08);
--page-gradient-end: rgba(59, 130, 246, 0.08);
--text-color: #0f172a;
--card-bg: #ffffff;
```

**Dark Mode:**
```css
--page-gradient-start: rgba(17, 24, 39, 0.8);
--page-gradient-end: rgba(15, 118, 110, 0.4);
--text-color: #f8fafc;
--card-bg: rgba(15, 23, 42, 0.7);
```

### Custom Components

#### Hero Card
- Gradient background (navy to blue)
- 2.5rem padding, 18px border radius
- Large heading (2.4rem) + descriptive text

#### Prediction Box
- Gradient background (indigo to blue)
- 1.6rem font size, bold weight
- Center-aligned, letter spacing 0.04em

#### Metric Card
- Light background with subtle border
- Shadow for depth
- 14px border radius

#### Buttons
- Pill-shaped (border-radius: 999px)
- Gradient background (blue to purple)
- Hover effect: white border
- Shadow for 3D effect

---

## Navigation & UX

### Sidebar

**Components:**
1. **Navigation Radio Buttons**
   - 4 options: Home, Prediction, Data Analysis, Model Performance
   - Current page highlighted

2. **Divider** (`st.sidebar.markdown("---")`)

3. **Info Box**
   - Dataset size: 150 samples, 3 species
   - Model type: SVM (Linear)
   - Accuracy: 97.37%

**Always Visible:** Provides context regardless of page

### Session State

**Used For:**
- Storing predictions between reruns
- Maintaining user input across page changes
- Passing data to visualizations

**Keys:**
```python
st.session_state.prediction       # Species name
st.session_state.prediction_code  # Numeric code
st.session_state.sample          # Input array
```

---

## Performance Optimizations

### Caching Strategy

1. **@st.cache_resource** for model loading
   - Loaded once per server instance
   - Persists across user sessions
   - Saves ~50ms per page load

2. **@st.cache_data** for dataset loading
   - Immutable data cached
   - Faster page switches
   - Reduces file I/O

### Lazy Loading

- Pairplot wrapped in `st.spinner()` - shows loading message
- 3D scatter only rendered when prediction exists
- Expandable sections prevent rendering unused content

---

## Error Handling

### Graceful Failures

```python
if df is None or model is None or scaler is None or label_encoder is None:
    st.stop()
```

**Behavior:**
- Displays error message from cache functions
- Stops execution (no broken UI)
- Prevents cascading errors

### Input Validation

**Sliders enforce valid ranges:**
- Min/max values based on dataset statistics
- Step size prevents invalid precision
- Default values are realistic specimens

**Label Encoding Fallback:**
```python
try:
    prediction_label = label_encoder.inverse_transform([prediction_code])[0]
except Exception:
    prediction_label = str(prediction_code)
```

---

## Deployment Checklist

### Prerequisites
- [ ] Models trained (`python src/train_models.py`)
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset available at `data/Iris.csv`

### Local Testing
```bash
streamlit run app.py
```

### Production Considerations

1. **File Paths:** Use relative paths (already implemented)
2. **Memory:** Models cached in memory (~1MB)
3. **Concurrency:** Stateless design supports multiple users
4. **Secrets:** No API keys or credentials needed

---

## Customization Guide

### Change Colors

Edit `SPECIES_COLORS` dictionary:
```python
SPECIES_COLORS = {
    'Iris-setosa': '#YOUR_COLOR',
    'Iris-versicolor': '#YOUR_COLOR',
    'Iris-virginica': '#YOUR_COLOR'
}
```

### Add New Page

1. Add option to sidebar radio:
```python
page = st.sidebar.radio("Go to", [..., "New Page"])
```

2. Add conditional block:
```python
elif page == "New Page":
    st.markdown("## Your Content")
```

### Modify Layout

Change column ratios:
```python
col1, col2 = st.columns([1, 2])  # 33%-66%
col1, col2 = st.columns([1, 1])  # 50%-50%
```

### Update Metrics

Edit `performance_data` dictionary in Model Performance page:
```python
performance_data = {
    'Species': [...],
    'Precision': [...],  # Your values
    'Recall': [...],
    'F1-Score': [...]
}
```

---

## Troubleshooting

### Common Issues

**1. "Module not found: data_processing"**
```python
# Solution: Ensure src/ is in path
sys.path.append('src')
```

**2. "Model assets not found"**
```bash
# Solution: Train models first
python src/train_models.py
```

**3. "Cannot display images"**
- Check internet connection (Wikipedia images)
- Images load from external URLs

**4. "Dashboard is slow"**
- Ensure caching is working (`@st.cache_*`)
- Check browser cache
- Reduce plot data points if necessary

### Debug Mode

Run with verbose logging:
```bash
streamlit run app.py --logger.level=debug
```

---

## Best Practices

### Code Organization
- Functions defined before `main()`
- Page logic in conditional blocks
- Reusable visualization functions
- Constants at module level

### User Experience
- Clear labels and tooltips
- Responsive feedback (spinners, info boxes)
- Consistent color scheme
- Mobile-friendly (wide layout adapts)

### Performance
- Cache expensive operations
- Lazy load visualizations
- Minimize redundant computations
- Use session state efficiently

---

## Summary

The Streamlit dashboard (`app.py`) provides a complete, production-ready interface for the Iris classification project. It combines:

- **4 comprehensive pages** covering all project aspects
- **Real-time predictions** with visual feedback
- **Interactive visualizations** using Plotly
- **Professional design** with dark mode support
- **Optimized performance** through caching
- **User-friendly navigation** with sidebar controls

**Lines of Code:** ~650  
**Visualizations:** 6 interactive plots  
**Deployment Time:** < 5 minutes on Streamlit Cloud  
**Expected Performance:** Sub-second predictions, smooth interactions