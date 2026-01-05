# Isolation Forest (iForest)

**Isolation Forest** is an **unsupervised anomaly (outlier) detection algorithm**.

>**Anomalies are easier to isolate than normal data points**

- Normal points → dense regions → need **many splits**
- Anomalies → rare & different → isolated in **few splits**

Isolation Forest uses random decision trees called **Isolation Trees (iTrees)**.

---

## Use Cases

- High-dimensional data  
- Large datasets  
- No clear density structure  

**Applications:**
- Fraud detection  
- Fault detection  
- Sensor data analysis  

---

## Working of Isolation Forest

For each isolation tree:

1. Randomly select **one feature**
2. Randomly select a **split value**
3. Split the data
4. Repeat until:
   - A point is isolated, or
   - Maximum depth is reached

> Each split uses **one feature**, but the full path uses **many features**.  
> Final isolation is a **combined effect**, not a single feature.

---

## Path Length

### `h(x)`

- `h(x)` = number of splits needed to isolate point `x` in **one tree**

Examples:
- Few splits → anomaly  
- Many splits → normal point  

---

## Average Path Length

### `E(h(x))`

- Average path length of point `x` across **all trees**

Interpretation:
- Small `E(h(x))` → fast isolation → **anomaly**
- Large `E(h(x))` → slow isolation → **normal**

---

## `c(n)`

**c(n) is the expected (theoretical) path length of a normal data point in a dataset of size n. and is constant and depends upon number of samples.**


## Anomaly Score

s(x, n) = 2 ^ ( - E(h(x)) / c(n) )


### Meaning of terms

- `x` → data point  
- `n` → number of samples  
- `E(h(x))` → average path length of `x`  
- `c(n)` → expected path length of a normal point  

---

## Interpretation

| Condition | Meaning |
|---------|--------|
| `E(h(x)) << c(n)` | Anomaly |
| `E(h(x)) ≈ c(n)` | Normal |
| `E(h(x)) >> c(n)` | Very normal |

### Score Values

| Score `s(x)` | Interpretation |
|--------------|---------------|
| ≈ 1 | Strong anomaly |
| ≈ 0.5 | Normal |
| < 0.5 | Very normal |

**Common threshold:**
- `s(x) ≥ 0.5` → anomaly  
- `s(x) < 0.5` → normal  

---


- No distance calculation  
- No density estimation  
- Works well in high dimensions  
- Fast: `O(n log n)`  
- Scalable and robust  

---

## sklearn Example

```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

model.fit(X)
pred = model.predict(X)
# -1 → anomaly
#  1 → normal
