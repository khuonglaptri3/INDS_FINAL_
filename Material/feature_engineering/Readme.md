# Feature Engineering — Adult Income Dataset

Thư mục này chứa toàn bộ pipeline tiền xử lý và tạo đặc trưng cho bộ dữ liệu Adult (UCI), phục vụ bài toán phân loại thu nhập (>50K / ≤50K) kết hợp phân tích công bằng (fairness analysis).

---

## Cấu trúc thư mục

```
feature_engineering/
├── 01_nonlinear_transforms/       # Biến đổi phi tuyến cho biến số
│   ├── README.md
│   ├── yeo_johnson_transform.py
│   └── robust_scaling.py
├── 02_categorical_encoding/       # Mã hoá biến phân loại
│   ├── README.md
│   ├── country_income_group.py
│   ├── occupation_group.py
│   ├── marital_status.py
│   └── other_encoding.py
├── 03_econometric_interactions/   # Đặc trưng tương tác kinh tế lượng
│   ├── README.md
│   ├── human_capital.py
│   ├── household_labour.py
│   └── net_capital.py
├── 04_fairness_interactions/      # Tương tác theo sắc tộc (fairness)
│   ├── README.md
│   ├── edu_by_race.py
│   ├── hours_by_race.py
│   └── capital_by_race.py
└── pipeline.py                    # Sklearn Pipeline tổng hợp
```

---

## Tổng quan các bước xử lý

### 1. Biến đổi phi tuyến (`01_nonlinear_transforms/`)

Hai biến tài chính `capital-gain` và `capital-loss` có phân phối cực kỳ lệch phải (phần lớn bằng 0, một nhóm nhỏ có giá trị rất lớn). Thay vì log transform truyền thống, pipeline áp dụng **Yeo-Johnson** — mở rộng từ Box-Cox, xử lý được cả giá trị bằng 0 và âm, với tham số λ tối ưu bằng MLE.

Các biến số còn lại (`age`, `hours-per-week`, `education-num`) được chuẩn hoá bằng **Robust Scaling** (dùng median và IQR thay vì mean/std) để giảm ảnh hưởng của ngoại lai.

### 2. Mã hoá biến phân loại (`02_categorical_encoding/`)

Thay vì mã hoá thô từng giá trị, pipeline thực hiện **nén ngữ nghĩa** trước khi mã hoá:

| Biến gốc | Xử lý | Phương pháp mã hoá |
|---|---|---|
| `native-country` (41 quốc gia) | Gom theo nhóm thu nhập World Bank → `country_income_group` | Ordinal Encoding |
| `occupation` (14 nghề) | Gom thành 4 nhóm kỹ năng | CatBoost Encoding |
| `workclass` | Giữ nguyên | Leave-One-Out Encoding |
| `marital-status` | Nhị phân hoá: đã kết hôn / chưa kết hôn | Binary (0/1) |
| `relationship`, `race` | Giữ nguyên | One-Hot Encoding |
| `sex` | Giữ nguyên | Binary (0/1) |

### 3. Đặc trưng tương tác kinh tế lượng (`03_econometric_interactions/`)

Ba đặc trưng được xây dựng dựa trên lý thuyết kinh tế lượng, không dựa trên quy tắc xã hội học:

| Ký hiệu | Công thức | Ý nghĩa |
|---|---|---|
| F1 | `age × education_num` | Tích luỹ vốn nhân lực theo thời gian |
| F2 | `hours_per_week × married_flag` | Xu hướng làm thêm giờ của người đã kết hôn |
| F3 | `capital_gain − capital_loss` | Sức mạnh tài chính ròng |

### 4. Tương tác theo sắc tộc (`04_fairness_interactions/`)

Ba đặc trưng điều kiện theo sắc tộc, phục vụ riêng cho phân tích fairness — định lượng sự bất bình đẳng kinh tế giữa các nhóm:

| Ký hiệu | Công thức | Ý nghĩa |
|---|---|---|
| F4 | `education_num × race` | Lợi tức giáo dục theo sắc tộc |
| F5 | `hours_per_week × race` | Gánh nặng giờ làm theo sắc tộc |
| F6 | `net_capital × race` | Khả năng tiếp cận vốn theo sắc tộc |

---

## Chạy pipeline

```python
from feature_engineering.pipeline import build_pipeline

pipeline = build_pipeline()
X_train_transformed = pipeline.fit_transform(X_train, y_train)
X_test_transformed  = pipeline.transform(X_test)
```

Pipeline được xây dựng bằng `sklearn.pipeline.Pipeline` và `ColumnTransformer`, đảm bảo không rò rỉ thông tin từ tập test vào quá trình fit (đặc biệt quan trọng với CatBoost Encoding và Leave-One-Out Encoding).

---

## Yêu cầu

```
scikit-learn >= 1.3
category_encoders >= 2.6
pandas >= 2.0
numpy >= 1.24
scipy >= 1.11
```

---

## Tài liệu tham khảo

- Yeo, I. K., & Johnson, R. A. (2000). A new family of power transformations to improve normality or symmetry. *Biometrika*, 87(4), 954–959.
- World Bank Country and Lending Groups: https://datahelpdesk.worldbank.org/knowledgebase/articles/906519
- Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality categorical attributes. *ACM SIGKDD Explorations*, 3(1), 27–32. (Leave-One-Out Encoding)