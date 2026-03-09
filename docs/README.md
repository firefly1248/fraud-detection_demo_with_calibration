# Documentation Assets

This directory contains documentation assets like images and diagrams.

## Generating Calibration Comparison Plot

To generate `calibration_comparison.png`, run the calibration comparison cells in `build_and_evaluate_model.ipynb`:

### Steps:

1. **Open the notebook:**
   ```bash
   jupyter notebook build_and_evaluate_model.ipynb
   ```

2. **Run the calibration comparison cells** (towards the end of the notebook):
   - Cells that train models with different calibration methods (isotonic, Venn-ABERS, sigmoid)
   - Cells that plot calibration curves
   - Cell that compares Expected Calibration Error (ECE)

3. **Save the calibration curve plot:**
   ```python
   import matplotlib.pyplot as plt

   # After generating the calibration comparison plot
   plt.savefig('docs/calibration_comparison.png', dpi=300, bbox_inches='tight')
   ```

4. **Verify the image:**
   ```bash
   ls -lh docs/calibration_comparison.png
   ```

### Alternative: Quick Script

You can also run this quick script to generate a comparison plot:

```python
from src.data_loader import load_fraud_data
from src.model import CalibratedBinaryClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Load data
df = load_fraud_data(sample_frac=0.1)
X = df.drop(columns=['isFraud', 'TransactionID'])
y = df['isFraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train models with different calibration methods
methods = ['isotonic', 'venn_abers', 'sigmoid']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, method in enumerate(methods):
    model = CalibratedBinaryClassifier(
        variable_params={'classifier__n_estimators': 100},
        calibration_method=method
    )
    model.fit(X_train, y_train)

    if method == 'venn_abers':
        y_pred = model.predict_proba_with_intervals(X_test)['p_combined']
    else:
        y_pred = model.predict_proba(X_test)[:, 1]

    # Plot calibration curve
    prob_true, prob_pred = calibration_curve(y_test, y_pred, n_bins=10)

    axes[idx].plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    axes[idx].plot(prob_pred, prob_true, 's-', label=f'{method.title()}')
    axes[idx].set_xlabel('Predicted probability')
    axes[idx].set_ylabel('True probability')
    axes[idx].set_title(f'{method.title()} Calibration')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('docs/calibration_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Calibration comparison plot saved to docs/calibration_comparison.png")
```

## Other Documentation Assets

- `architecture.png` - System architecture diagram (optional, create with draw.io or similar)
- `performance_benchmarks.png` - Performance metrics visualization (optional)
- `uncertainty_distribution.png` - Venn-ABERS interval width distribution (optional)

## Tips

- Use `dpi=300` for high-quality images suitable for GitHub README
- Use `bbox_inches='tight'` to remove excess whitespace
- Keep image file sizes reasonable (<1MB) for faster page loading
