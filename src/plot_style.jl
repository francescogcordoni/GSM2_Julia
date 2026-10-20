# ============================================================================
# src/plot_style.jl
#
# Common style of all figures of the paper:
#   boxed frame, light grid background, Computer Modern font, 600 dpi,
#   LaTeX labels for ions and symbols.
#
# Usage (any script in example/ or scripts/):
#   include(joinpath(@__DIR__, "..", "src", "plot_style.jl"))
#   ... default(...) of the script (sizes, margins) ...
#   paper_style!()                         # applied last: frame, grid, font, dpi
#
#   ion_label("12C_80MeVu")  → $^{12}$C 80 MeV/u   (LaTeXString)
#   tex("β (Gy\$^{-2}\$)")   → LaTeXString if the text contains $…$, else unchanged
# ============================================================================
using Plots, LaTeXStrings

const PAPER_STYLE = (framestyle = :box, grid = true, gridalpha = 0.3, gridlinewidth = 0.5,
                     dpi = 600, fontfamily = "Computer Modern")

"""Apply the paper style on top of the current Plots defaults (sizes and margins are kept)."""
paper_style!() = default(; PAPER_STYLE...)

"""Wrap text containing LaTeX math (\$…\$) as a LaTeXString, so it is rendered as LaTeX."""
tex(s::LaTeXString) = s
tex(s::AbstractString) = occursin('$', s) ? LaTeXString(s) : String(s)

"""
LaTeX label of a condition tag, e.g. "12C_80MeVu" → \$^{12}\$C 80 MeV/u,
"1H_100MeV" → \$^{1}\$H 100 MeV, "12C" → \$^{12}\$C, "control" → Control.
"""
function ion_label(tag)
    t = string(tag)
    lowercase(t) == "control" && return "Control"
    m = match(r"^(\d+)([A-Z][a-z]?)(?:_(\d+(?:\.\d+)?)\s*(MeVu|MeV))?$", t)
    m === nothing && return t
    A, el, E, unit = m.captures
    s = "\$^{$A}\$$el"
    E !== nothing && (s *= " $E " * (unit == "MeVu" ? "MeV/u" : "MeV"))
    return LaTeXString(s)
end

# Colours shared by the figures
const ION_COLOURS = Dict("control" => :gray40, "1H_100MeV" => :royalblue, "1H_30MeV" => :deepskyblue,
                         "1H_80MeV" => :royalblue, "12C_80MeVu" => :firebrick, "12C_10MeVu" => :darkorange)
const PHASE_COLOURS = Dict("G0" => :gray, "G1" => :steelblue, "S" => :seagreen,
                           "G2" => :darkorange, "M" => :firebrick)
