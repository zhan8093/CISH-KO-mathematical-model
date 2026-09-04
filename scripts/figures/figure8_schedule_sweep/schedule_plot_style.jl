using Plots

# Shared plotting style aligned with the figures in updates_12_4.
const CISH_FIGURE_COLORS = [
    RGB(0.180, 0.525, 0.671),  # steel blue
    RGB(0.910, 0.333, 0.227),  # warm red
    RGB(0.498, 0.690, 0.412),  # muted green
    RGB(0.961, 0.651, 0.137),  # amber
    RGB(0.557, 0.267, 0.678),  # purple
    RGB(0.204, 0.286, 0.369),  # dark slate
]

function apply_cish_plot_style!(; size = (520, 420))
    gr(fontfamily = "Computer Modern")
    default(
        framestyle = :axes,
        grid = true,
        gridalpha = 0.3,
        gridstyle = :dash,
        guidefontsize = 14,
        tickfontsize = 11,
        titlefontsize = 12,
        legendfontsize = 9,
        background_color = :white,
        foreground_color = :black,
        dpi = 300,
        size = size,
        margin = 5Plots.mm,
    )
    return nothing
end

function save_png_pdf(plt, png_path::AbstractString)
    savefig(plt, png_path)
    root, _ = splitext(png_path)
    savefig(plt, root * ".pdf")
    return nothing
end
