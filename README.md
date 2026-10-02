# Semantic Cache Rust

> Motor de cache semântico de baixa latência (<2ms) e tokenização SIMD para redução de custos com OpenRouter via servidor MCP nativo em Rust.

## Visão Geral

O **Semantic Cache Rust** é o componente de aceleração vetorial e otimização de custos do ecossistema. Construído em Rust para garantir zero garbage collection e máxima previsibilidade de latência, ele intercepta consultas antes da chamada ao OpenRouter e avalia similaridade semântica em memória.

## Como Funciona

```
[ Prompt do Usuário ] ──▶ [ Semantic Cache Rust ]
                                 │
                  ┌──────────────┴──────────────┐
                  ▼                             ▼
       [ Similaridade >= 0.92 ]      [ Cache Miss (< 0.92) ]
                  │                             │
                  ▼                             ▼
      Retorna Cache (< 2ms)         Repassa ao OpenRouter
   (Economia de até 80% tokens)    (Salva novo embedding no cache)
```

1. **Similaridade SIMD:** Cálculo acelerado por hardware de similaridade de cosseno e produto escalar sobre embeddings em memória.
2. **Tokenização Nativa:** Contagem precisa e ultrarrápida de tokens para dimensionamento de contexto sem depender de APIs externas.
3. **Servidor MCP:** Expõe ferramentas (tools) para verificação de cache, invalidação semântica e métricas de economia de tokens para clientes MCP (Claude, Cursor, Studio).

## Arquitetura & Módulos

O desenvolvimento é guiado pelo roadmap de **100 PRs** no [Plane da in100tiva (CACHRS)](https://plane.in100tiva.com/in100tiva/):

- **Fase 1:** Fundação, CI/CD e Contratos (PRs 01-20)
- **Fase 2:** Core Engine e Protocolo MCP (PRs 21-50)
- **Fase 3:** Adaptadores, Conectores e Streaming (PRs 51-75)
- **Fase 4:** Observabilidade OTel e Benchmarks de Latência (PRs 76-90)
- **Fase 5:** Release v1.0, Docker e Documentação (PRs 91-100+)

## Regras de Engenharia

- **Tamanho dos PRs:** Mínimo 100 linhas, máximo 500 linhas de código.
- **Commits:** Atômicos seguindo padrão Conventional Commits.
- **Contract-First:** Schemas e interfaces definidos primeiro para trabalho paralelo sem bloqueios mútuos.

## Mantenedores

- **Luan Oliveira** ([@in100tiva](https://github.com/in100tiva)) — Arquiteto de Software & Tech Lead
- **Victor Nascimento** ([@VictorNascimento14](https://github.com/VictorNascimento14)) — Tech Lead & Engenheiro de Software
