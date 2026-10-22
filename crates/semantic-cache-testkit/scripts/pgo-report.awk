# Relatório do scripts/pgo.sh: as rodadas do bench `ipc` com e sem PGO.
#   LC_ALL=C awk -v expect=<rodadas> -f scripts/pgo-report.awk <dir>/base-1.txt <dir>/pgo-1.txt ...
# Cada arquivo é a saída de uma rodada, <base|pgo>-<rodada>.txt. De cada linha da
# tabela do bench saem o p50 de ponta a ponta e o p50 no servidor; em cada rodada,
# Δ = (pgo − base) / base. Sai a mediana dos p50 de cada versão e a dos Δ, com o
# número de rodadas em que o PGO foi mais rápido: uma rodada perturbada pesa uma
# posição na mediana e 1 na contagem, e numa faixa mín–máx decidiria sozinha.
# Linha faltando ou repetida, número ilegível e rodada incompleta são erro (saída
# 1, nada no stdout), nunca uma tabela pela metade. Só awk POSIX.
# O LC_ALL=C é obrigatório: o mawk lê e escreve número pelo LC_NUMERIC, e em pt_BR
# "0.084" vira 0 sem erro nenhum (o BEGIN recusa).

function fail(msg) {
    print "pgo-report: " msg > "/dev/stderr"
    failed = 1
    exit 1
}

function trim(s) {
    gsub(/^ +| +$/, "", s)
    return s
}

# Mediana dos `n` primeiros valores de `list` (inserção numa cópia ordenada).
function median(list, n,    i, j, x, sorted) {
    for (i = 1; i <= n; i++) {
        x = list[i]
        for (j = i - 1; j >= 1 && sorted[j] > x; j--)
            sorted[j + 1] = sorted[j]
        sorted[j + 1] = x
    }
    return n % 2 ? sorted[(n + 1) / 2] : (sorted[n / 2] + sorted[n / 2 + 1]) / 2
}

# "base → pgo | Δ (rodadas mais rápidas)" de uma coluna da linha `key` (1 ponta a
# ponta, 2 no servidor). Base 0 não tem Δ; sem Δ nenhum, "—".
function compare(key, col,    r, nd, faster, b, p, d) {
    nd = faster = 0
    for (r = 1; r <= expect; r++) {
        if (!(("base", r, key, col) in value) || !(("pgo", r, key, col) in value))
            fail("a rodada " r " não tem a linha " key " nas duas versões")
        b[r] = value["base", r, key, col]
        p[r] = value["pgo", r, key, col]
        if (b[r] > 0) {
            d[++nd] = (p[r] - b[r]) / b[r] * 100
            faster += p[r] < b[r]
        }
    }
    return sprintf("%.3f → %.3f ms | %s", median(b, expect), median(p, expect),
        nd ? sprintf("%+.1f%% (%d/%d)", median(d, nd), faster, nd) : "—")
}

BEGIN {
    if (expect !~ /^[1-9][0-9]*$/) fail("uso: awk -v expect=<rodadas> -f pgo-report.awk <arquivos>")
    if ("0.5" + 0 != 0.5 || sprintf("%.1f", 0.5) != "0.5") fail("o separador decimal não é o ponto: rode com LC_ALL=C")
}

FNR == 1 {
    if (!match(FILENAME, /(base|pgo)-[1-9][0-9]*\.txt$/))
        fail("arquivo fora do padrão <base|pgo>-<rodada>.txt: " FILENAME)
    split(substr(FILENAME, RSTART, RLENGTH - 4), name, "-")
    variant = name[1]
    round = name[2] + 0
}

/^\| flat_/ {
    split($0, cell, "|")
    key = trim(cell[2]) " | " trim(cell[3])
    split(cell[4], e2e, " ")
    split(cell[5], srv, " ")
    if (e2e[1] !~ /^[0-9]+\.[0-9]+$/ || srv[1] !~ /^[0-9]+\.[0-9]+$/)
        fail("número ilegível em " FILENAME ": " $0)
    if ((variant, round, key, 1) in value)
        fail("linha repetida em " FILENAME ": " key)
    value[variant, round, key, 1] = e2e[1] + 0
    value[variant, round, key, 2] = srv[1] + 0
    if (!(key in seen)) {
        seen[key] = 1
        order[++keys] = key
    }
}

END {
    if (failed) exit 1
    if (!keys) fail("nenhuma linha da tabela do bench")
    for (k = 1; k <= keys; k++)
        row[k] = "| " order[k] " | " compare(order[k], 1) " | " compare(order[k], 2) " |"
    print "| índice | vetores | ponta a ponta p50: base → PGO | Δ (rodadas) | no servidor p50: base → PGO | Δ (rodadas) |"
    print "|---|---|---|---|---|---|"
    for (k = 1; k <= keys; k++)
        print row[k]
}
