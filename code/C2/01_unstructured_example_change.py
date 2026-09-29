from unstructured.partition.pdf import partition_pdf
from collections import Counter

# PDF 文件路径
pdf_path = "../../data/C2/pdf/rag.pdf"


# ==================== 1. 使用 hi_res 策略解析 ====================

# hi_res：高分辨率解析策略
# 更关注 PDF 的页面布局和文档结构，
# 适合包含标题、正文、表格、图片等复杂版式的 PDF。
hi_res_elements = partition_pdf(
    filename=pdf_path,
    strategy="hi_res"
)

# 输出解析结果的基本统计信息
print("\n========== hi_res 解析结果 ==========")
print(
    f"解析完成: {len(hi_res_elements)} 个元素, "
    f"{sum(len(str(e)) for e in hi_res_elements)} 字符"
)

# 统计解析出的元素类型
# 例如 Title、NarrativeText、Text、Table 等
hi_res_types = Counter(e.category for e in hi_res_elements)
print(f"元素类型: {dict(hi_res_types)}")

# 显示所有解析元素
for i, element in enumerate(hi_res_elements, 1):
    print(f"\nElement {i} ({element.category}):")
    print(element)
    print("=" * 60)


# ==================== 2. 使用 ocr_only 策略解析 ====================

# ocr_only：主要通过 OCR 识别 PDF 页面中的文字
# 更适合扫描版 PDF、图片型 PDF 等没有可直接提取文本层的文档。
ocr_elements = partition_pdf(
    filename=pdf_path,
    strategy="ocr_only"
)

# 输出解析结果的基本统计信息
print("\n========== ocr_only 解析结果 ==========")
print(
    f"解析完成: {len(ocr_elements)} 个元素, "
    f"{sum(len(str(e)) for e in ocr_elements)} 字符"
)

# 统计元素类型
ocr_types = Counter(e.category for e in ocr_elements)
print(f"元素类型: {dict(ocr_types)}")

# 显示所有解析元素
for i, element in enumerate(ocr_elements, 1):
    print(f"\nElement {i} ({element.category}):")
    print(element)
    print("=" * 60)