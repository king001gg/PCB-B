"""数据导出模块。

支持将检测结果导出为多种格式：
    - CSV（通用表格）
    - Excel (.xlsx)
    - JSON（程序间交换）
    - PDF 报告（需 fpdf2 或 reportlab）
"""

import os
import csv
import json
from typing import List, Optional
from datetime import datetime

from core.quality import QualityReport
from core.defects import Defect


def _rnd(value, digits: int):
    """None 安全的 rounding。

    色度类字段可以是 None（未测 / 不可测），必须原样保留成 None 让表格留空，
    不能变成 0 —— 0 会被读成「色度零偏移」即满分。
    """
    return None if value is None else round(value, digits)


def export_csv(
    reports: List[QualityReport],
    output_path: str,
    include_header: bool = True,
) -> None:
    """导出质量报告为 CSV 文件。

    Args:
        reports: 质量报告列表。
        output_path: 输出文件路径。
        include_header: 是否包含列标题。
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if include_header:
            writer.writerow([
                "board_id", "timestamp", "overall_score", "ok_ng",
                "roughness_uniformity", "roughness_std",
                "direction_consistency", "oxidation_percentage",
                "embedding_count", "unroughened_percentage",
                "warnings",
                # 色度 / 饱和度，**追加在末尾**：前 11 列的位置不得变动，
                # 否则下游按列号解析的脚本会静默错位。
                "color_available", "color_hue_mean_deg",
                "color_hue_deviation_deg", "color_sat_mean",
                "color_oor_abs_pct", "color_oor_adaptive_pct",
                "color_oor_count",
            ])
        for r in reports:
            writer.writerow([
                r.board_id, r.timestamp, r.overall_score,
                1 if r.ok_ng else 0, r.roughness_uniformity,
                r.roughness_std, r.direction_consistency,
                r.oxidation_percentage, r.embedding_count,
                r.unroughened_percentage,
                "; ".join(r.warnings),
                1 if r.color_available else 0,
                r.color_hue_mean_deg,
                r.color_hue_deviation_deg,
                r.color_sat_mean,
                r.color_oor_abs_pct,
                r.color_oor_adaptive_pct,
                r.color_oor_count,
            ])

    print(f"[Export] CSV 已导出: {output_path} ({len(reports)} 条记录)")


def export_excel(
    reports: List[QualityReport],
    defects_list: List[List[Defect]] = None,
    output_path: str = None,
) -> None:
    """导出为 Excel 文件（需要 openpyxl）。

    包含两个工作表：
        - 质量报告（每条检测一行）
        - 缺陷详情（每条缺陷一行）
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment
    except ImportError:
        print("[WARN] openpyxl 未安装，回退到 CSV 导出")
        export_csv(reports, output_path or "report.csv")
        return

    wb = Workbook()

    # --- 工作表 1: 质量报告 ---
    ws = wb.active
    ws.title = "质量报告"

    # 标题
    headers = [
        "板号", "时间", "总分", "判定", "粗糙度均匀性",
        "粗糙度标准差", "方向一致性", "氧化面积%",
        "磨料嵌入数", "未粗化面积%", "预警",
        # 色度 / 饱和度，追加在末尾（前 11 列位置不变）
        "色度偏移(°)", "色相均值(°)", "饱和度",
        "越界面积%(绝对)", "越界面积%(自适应)", "越界区域数",
        "色度已测",
    ]
    for col, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=h)
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color="4472C4",
                                 end_color="4472C4", fill_type="solid")
        cell.font = Font(bold=True, color="FFFFFF")

    for row, r in enumerate(reports, 2):
        ws.cell(row=row, column=1, value=r.board_id)
        ws.cell(row=row, column=2, value=r.timestamp)
        ws.cell(row=row, column=3, value=r.overall_score)
        ws.cell(row=row, column=4, value="OK" if r.ok_ng else "NG")
        ws.cell(row=row, column=5, value=r.roughness_uniformity)
        ws.cell(row=row, column=6, value=round(r.roughness_std, 4))
        ws.cell(row=row, column=7, value=round(r.direction_consistency, 4))
        ws.cell(row=row, column=8, value=round(r.oxidation_percentage, 2))
        ws.cell(row=row, column=9, value=r.embedding_count)
        ws.cell(row=row, column=10, value=round(r.unroughened_percentage, 2))
        ws.cell(row=row, column=11, value="; ".join(r.warnings))
        # 色度 / 饱和度（只监测，不进总分）。None 就是空单元格 —— 不填 0。
        ws.cell(row=row, column=12, value=_rnd(r.color_hue_deviation_deg, 2))
        ws.cell(row=row, column=13, value=_rnd(r.color_hue_mean_deg, 2))
        ws.cell(row=row, column=14, value=_rnd(r.color_sat_mean, 2))
        ws.cell(row=row, column=15, value=round(r.color_oor_abs_pct, 4))
        ws.cell(row=row, column=16, value=round(r.color_oor_adaptive_pct, 4))
        ws.cell(row=row, column=17, value=r.color_oor_count)
        ws.cell(row=row, column=18, value="是" if r.color_available else "否")

        # NG 行标红
        if not r.ok_ng:
            for col in range(1, len(headers) + 1):
                ws.cell(row=row, column=col).font = Font(color="FF0000")

    # --- 工作表 2: 缺陷详情 ---
    if defects_list:
        ws2 = wb.create_sheet("缺陷详情")
        def_headers = [
            "板号", "缺陷类型", "面积(mm²)", "严重度",
            "置信度", "X1", "Y1", "X2", "Y2",
        ]
        for col, h in enumerate(def_headers, 1):
            cell = ws2.cell(row=1, column=col, value=h)
            cell.font = Font(bold=True)

        row = 2
        for i, defects in enumerate(defects_list):
            board_id = reports[i].board_id if i < len(reports) else ""
            for d in defects:
                x1, y1, x2, y2 = d.bbox
                ws2.cell(row=row, column=1, value=board_id)
                ws2.cell(row=row, column=2, value=d.type)
                ws2.cell(row=row, column=3, value=round(d.area_mm2, 4))
                ws2.cell(row=row, column=4, value=round(d.severity, 2))
                ws2.cell(row=row, column=5, value=round(d.confidence, 2))
                ws2.cell(row=row, column=6, value=x1)
                ws2.cell(row=row, column=7, value=y1)
                ws2.cell(row=row, column=8, value=x2)
                ws2.cell(row=row, column=9, value=y2)
                row += 1

    # 自动列宽
    for ws in [wb.active] + wb.worksheets[1:]:
        for col in ws.columns:
            max_len = 0
            for cell in col:
                if cell.value:
                    max_len = max(max_len, len(str(cell.value)))
            ws.column_dimensions[col[0].column_letter].width = max_len + 2

    wb.save(output_path)
    print(f"[Export] Excel 已导出: {output_path} ({len(reports)} 条)")


def export_json(
    reports: List[QualityReport],
    defects_list: List[List[Defect]] = None,
    output_path: str = None,
) -> None:
    """导出为 JSON 文件。"""
    data = []
    for i, r in enumerate(reports):
        entry = r.to_dict()
        if defects_list and i < len(defects_list):
            entry["defects"] = [
                {
                    "type": d.type,
                    "area_pixels": d.area_pixels,
                    "area_mm2": d.area_mm2,
                    "centroid": list(d.centroid),
                    "bbox": list(d.bbox),
                    "severity": d.severity,
                    "confidence": d.confidence,
                }
                for d in defects_list[i]
            ]
        data.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[Export] JSON 已导出: {output_path} ({len(reports)} 条)")


def export_pdf(
    report: QualityReport,
    defects: List[Defect] = None,
    output_path: str = None,
) -> None:
    """导出为 PDF 检测报告（需要 fpdf2）。"""
    try:
        from fpdf import FPDF
    except ImportError:
        print("[WARN] fpdf2 未安装。安装: pip install fpdf2")
        return

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # 标题
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "PCB 喷砂表面质量检测报告", ln=True, align="C")
    pdf.ln(5)

    # 基本信息
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"板号: {report.board_id}", ln=True)
    pdf.cell(0, 6, f"检测时间: {report.timestamp}", ln=True)
    pdf.ln(3)

    # 判定
    status = "OK - 合格" if report.ok_ng else "NG - 不合格"
    pdf.set_font("Helvetica", "B", 14)
    if report.ok_ng:
        pdf.set_text_color(0, 128, 0)
    else:
        pdf.set_text_color(255, 0, 0)
    pdf.cell(0, 10, status, ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    # 评分
    pdf.set_font("Helvetica", "B", 36)
    pdf.cell(0, 20, f"{report.overall_score:.1f}", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "综合质量评分 (0-100)", ln=True, align="C")
    pdf.ln(5)

    # 分项指标表
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "分项指标", ln=True)
    pdf.set_font("Helvetica", "", 10)

    metrics = [
        ("粗糙度均匀性", f"{report.roughness_uniformity:.1f}/100"),
        ("粗糙度 CV", f"{report.roughness_std:.4f}"),
        ("方向一致性", f"{report.direction_consistency:.4f}"),
        ("氧化斑面积", f"{report.oxidation_percentage:.2f}%"),
        ("磨料嵌入", f"{report.embedding_count} 个"),
        ("未粗化面积", f"{report.unroughened_percentage:.2f}%"),
    ]

    # 色度 / 饱和度（只监测，不进总分）
    if not report.color_available:
        metrics.append(("色度偏移", "未测（灰度输入）"))
    else:
        hue_dev = report.color_hue_deviation_deg
        hue_mean = report.color_hue_mean_deg
        sat = report.color_sat_mean
        metrics.append(("色度偏移", "--" if hue_dev is None
                        else f"{hue_dev:.1f}°"))
        metrics.append(("色相均值", "--" if hue_mean is None
                        else f"{hue_mean:.1f}°"))
        metrics.append(("饱和度", "--" if sat is None else f"{sat:.1f}"))
        metrics.append(("越界面积(绝对)", f"{report.color_oor_abs_pct:.2f}%"))
        metrics.append(("越界面积(自适应)",
                        f"{report.color_oor_adaptive_pct:.2f}%"))
        metrics.append(("越界区域数", f"{report.color_oor_count} 处"))
    for label, value in metrics:
        pdf.cell(60, 6, label)
        pdf.cell(0, 6, value, ln=True)

    pdf.ln(3)

    # 缺陷列表
    if defects:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, f"检测到的缺陷 ({len(defects)} 处)", ln=True)
        pdf.set_font("Helvetica", "", 9)
        for d in defects:
            x1, y1, x2, y2 = d.bbox
            line = (
                f"  [{d.type}] 面积={d.area_mm2:.2f}mm² "
                f"严重度={d.severity:.2f} 位置=({x1},{y1})-({x2},{y2})"
            )
            pdf.cell(0, 5, line, ln=True)

    # 预警
    if report.warnings:
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(255, 0, 0)
        pdf.cell(0, 8, "预警信息", ln=True)
        pdf.set_font("Helvetica", "", 9)
        for w in report.warnings:
            pdf.cell(0, 5, f"  ! {w}", ln=True)

    pdf.output(output_path)
    print(f"[Export] PDF 已导出: {output_path}")
