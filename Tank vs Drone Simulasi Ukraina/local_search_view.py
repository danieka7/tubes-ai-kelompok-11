import pygame

from display_config import COLORS
from labels import ACTION_LABELS


# -----------------------------------------------------------------------
# Visualisasi local search: grafik konvergensi (skor rencana yang dipegang vs
# skor terbaik per iterasi) + rencana terbaik per ronde. Menggantikan pohon
# pencarian saat mode NPC adalah Hill Climbing / Simulated Annealing.
# -----------------------------------------------------------------------
LOCAL_CHART_HEIGHT_RATIO = 0.45   # porsi tinggi area untuk grafik
LOCAL_CHART_MIN_HEIGHT = 70
LOCAL_CHART_LEFT_MARGIN = 52      # ruang label sumbu-Y
LOCAL_CHART_MIN_SPAN = 1.0        # cegah pembagian nol saat semua skor sama
LOCAL_CHART_TOP_OFFSET = 24       # ruang legenda di atas grafik
LOCAL_CHART_INSET = 1             # garis tepi grafik 1 px, supaya polyline tidak menimpanya
LOCAL_CHART_LINE_WIDTH = 2
LOCAL_CAPTION_GAP = 2
LOCAL_VIEW_LINE_H = 16
LOCAL_VIEW_SECTION_GAP = 6
LOCAL_VIEW_MESSAGE_OFFSET = 20
LOCAL_LEGEND_SWATCH = 10
LOCAL_LEGEND_SWATCH_TOP = 3
LOCAL_LEGEND_TEXT_GAP = 16
LOCAL_LEGEND_ITEM_GAP = 14
LOCAL_VIEW_BG = (20, 29, 17)


def _chart_points(trace, rect, low, high):
    """Ubah deret skor menjadi titik layar polyline di dalam `rect`.

    Kompleksitas: waktu O(n), memori O(n).
    """
    span = max(high - low, LOCAL_CHART_MIN_SPAN)
    last = max(1, len(trace) - 1)
    usable_w = rect.w - LOCAL_CHART_INSET
    usable_h = rect.h - 2 * LOCAL_CHART_INSET
    return [
        (rect.x + usable_w * index / last, rect.bottom - LOCAL_CHART_INSET - usable_h * (value - low) / span)
        for index, value in enumerate(trace)
    ]


def _plan_lines(agent):
    """Teks rencana terbaik per ronde, mis. 'Putaran +0: Serang'.

    Kompleksitas: waktu O(horizon), memori O(horizon).
    """
    return [f"Putaran +{offset}: {ACTION_LABELS[kind]}" for offset, kind in enumerate(agent.best_plan)]


def _draw_chart_legend(screen, fonts, area):
    legend_x = area.x
    for label, color in (("skor rencana yang dipegang", COLORS["drone"]), ("skor terbaik", COLORS["accent_hover"])):
        pygame.draw.rect(
            screen, color,
            (legend_x, area.y + LOCAL_LEGEND_SWATCH_TOP, LOCAL_LEGEND_SWATCH, LOCAL_LEGEND_SWATCH),
            border_radius=2,
        )
        text = fonts["small"].render(label, True, COLORS["text_dim"])
        screen.blit(text, (legend_x + LOCAL_LEGEND_TEXT_GAP, area.y))
        legend_x += LOCAL_LEGEND_TEXT_GAP + text.get_width() + LOCAL_LEGEND_ITEM_GAP


def _draw_convergence_chart(screen, fonts, agent, area):
    """Gambar grafik konvergensi; kembalikan koordinat Y tepat di bawah grafik."""
    font_small = fonts["small"]
    _draw_chart_legend(screen, fonts, area)

    chart = pygame.Rect(
        area.x + LOCAL_CHART_LEFT_MARGIN, area.y + LOCAL_CHART_TOP_OFFSET,
        area.w - LOCAL_CHART_LEFT_MARGIN, max(LOCAL_CHART_MIN_HEIGHT, int(area.h * LOCAL_CHART_HEIGHT_RATIO)),
    )
    pygame.draw.rect(screen, LOCAL_VIEW_BG, chart, border_radius=4)
    pygame.draw.rect(screen, COLORS["panel_border"], chart, width=1, border_radius=4)

    low = min(min(agent.current_trace), min(agent.best_trace))
    high = max(max(agent.current_trace), max(agent.best_trace))
    screen.blit(font_small.render(f"{high:.0f}", True, COLORS["text_dim"]), (area.x, chart.y))
    screen.blit(font_small.render(f"{low:.0f}", True, COLORS["text_dim"]), (area.x, chart.bottom - LOCAL_VIEW_LINE_H))

    for trace, color in ((agent.current_trace, COLORS["drone"]), (agent.best_trace, COLORS["accent_hover"])):
        points = _chart_points(trace, chart, low, high)
        if len(points) >= 2:
            pygame.draw.lines(screen, color, False, points, LOCAL_CHART_LINE_WIDTH)

    caption = f"iterasi 1 .. {len(agent.best_trace)}"
    screen.blit(font_small.render(caption, True, COLORS["text_dim"]), (chart.x, chart.bottom + LOCAL_CAPTION_GAP))
    return chart.bottom + LOCAL_CAPTION_GAP + LOCAL_VIEW_LINE_H + LOCAL_VIEW_SECTION_GAP


def draw_local_search_view(screen, fonts, agent, area):
    """Gambar hasil keputusan local search terakhir di dalam `area` (pygame.Rect).

    Kompleksitas: waktu O(n + horizon), n = jumlah iterasi tercatat.
    """
    if not agent.best_trace:
        text = fonts["ui"].render(
            "Belum ada rencana local search -- tunggu NPC mengambil keputusan.", True, COLORS["text_dim"],
        )
        screen.blit(text, (area.x + LOCAL_VIEW_MESSAGE_OFFSET, area.y + LOCAL_VIEW_MESSAGE_OFFSET))
        return

    y = _draw_convergence_chart(screen, fonts, agent, area)
    title = f"Rencana terbaik {agent.root_state.npcSpec.name} (aksi per putaran):"
    screen.blit(fonts["ui_bold"].render(title, True, COLORS["text_dim"]), (area.x, y))
    y += LOCAL_VIEW_LINE_H + LOCAL_VIEW_SECTION_GAP

    max_lines = max(0, (area.bottom - y) // LOCAL_VIEW_LINE_H)
    for line in _plan_lines(agent)[:max_lines]:
        screen.blit(fonts["small"].render(line, True, COLORS["text_cream"]), (area.x, y))
        y += LOCAL_VIEW_LINE_H
