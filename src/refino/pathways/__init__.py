"""Predição de atividade de vias a partir dos picos, sem identificação prévia.

O pacote parte dos `RT_mz` que o laboratório tem e prediz atividade de via sem exigir que
alguém já tenha identificado os compostos. Ele nunca afirma identidade de composto: devolve
via com evidência agregada, e os nomes na coluna de sobreposição são candidatos, não
atribuições. Identificar um `RT_mz` como composto continua fora de escopo.

O enriquecimento de uma lista já identificada é outro modo: roda no R portátil, e o recorte
da entrada dele vive em `pathways/enriquecimento.py`.

Núcleo puro: não conhece banco, HTTP, sistema de arquivos nem R. Recebe nomes de feature e
vetores em memória e devolve estrutura em memória. Quem escreve arquivo é `io/`, e quem
desenha é a janela.
"""
