# Proposta curta — Instituto Centec / CVT Fortaleza

## Título de trabalho

**Dinâmica latente sob convergência observável em sistemas recorrentes**  
*Accumulating fixed points, dependência de trajetória e acoplamento temporal no projeto Demian*

## Proponente

Projeto independente de pesquisa e desenvolvimento em redes neurais recorrentes e sistemas dinâmicos.  
Plataforma experimental: **Demian**.

## Problema

Em sistemas recorrentes, uma saída ou superfície observável pode aparentar estabilidade sem que o estado interno completo tenha necessariamente estabilizado.

A pergunta central é:

> **Quando a convergência observável representa um ponto fixo real do sistema e quando ela esconde dinâmica interna persistente, dependente da história e relevante para trajetórias futuras?**

Formalmente, o projeto investiga regimes em que:

```text
G(z[t+1]) ~= G(z[t])
```

enquanto:

```text
z[t+1] != z[t]
```

onde `z` representa o estado recorrente e `G` a superfície/readout observável.

O nome de trabalho usado no laboratório para esse regime é **Accumulating Fixed Point (AFP)**. O termo não implica que o estado completo esteja em um ponto fixo matemático; trata-se de um ponto fixo **projetado/observável** com possível dinâmica latente.

## Origem do problema

O fenômeno apareceu durante uma sequência de experimentos que começou com baselines recorrentes convencionais e evoluiu por várias arquiteturas Demian.

A linhagem experimental selecionada para esta investigação é:

```text
RNN / GRU / LSTM
        |
native v2 -> v3 -> v8 -> v9 -> v9 five-channel -> Demian v1
```

Os experimentos históricos mostraram que classificações de superfície do tipo `FIXED_POINT` podiam coexistir com estrutura interna em canais recorrentes. O v9 five-channel introduziu explicitamente `message` e `carrier`; trabalhos posteriores de ablação e continuidade levaram ao Demian v1, atualmente composto por:

```text
fast -> message -> carrier -> slow
          \          /
           -> gate <-
              |
       modulação de rotas
```

com os canais `fast`, `slow`, `control`, `message`, `carrier` e `gate`.

## Hipóteses testáveis

### H1 — Separação superfície/estado

Existem arquiteturas recorrentes em que a velocidade da superfície observável se aproxima de zero enquanto o estado recorrente completo mantém movimento mensurável.

### H2 — Dependência de trajetória

Estados com superfícies muito semelhantes, mas interiores diferentes, podem produzir continuações diferentes sob o mesmo estímulo futuro.

### H3 — Efeito da arquitetura

A presença, intensidade e forma dessa separação muda sistematicamente quando a arquitetura altera:

- quantidade de canais internos;
- exposição do hidden state;
- escalas temporais;
- roteamento entre canais;
- mecanismos de gate/controle.

### H4 — Generalização de acoplamento

O mesmo substrato recorrente pode atuar como observador de diferentes sistemas dinâmicos desde que exista uma sequência ordenada de medições:

```text
observação S(t) -> adaptador phi -> input recorrente x[t] -> trajetória interna z[t]
```

O índice `t` pode representar tempo físico, frames, janelas de EEG, profundidade em perfil geológico, passos de simulação ou outra ordem temporal/causal apropriada ao domínio.

## Metodologia

### Experimento 1 — Linha arquitetural

Comparar, com protocolo reproduzível:

- RNN;
- GRU;
- LSTM;
- Demian native v2;
- native v3;
- native v8;
- native v9;
- v9 five-channel;
- Demian v1.

Métricas principais:

- velocidade RMS da superfície;
- velocidade RMS do estado recorrente completo;
- razão movimento latente / movimento superficial;
- comprimento de trajetória;
- dimensão da superfície versus dimensão latente;
- classificação de atrator;
- sensibilidade a perturbações;
- prevalência do regime AFP sob critérios definidos previamente.

### Experimento 2 — Consequência causal

Selecionar estados:

```text
surface(A) ~= surface(B)
latent(A) != latent(B)
```

Aplicar a mesma continuação ou perturbação e medir:

```text
future(A) versus future(B)
```

No Demian, complementar com:

- state surgery;
- channel ablation;
- full capsule resume;
- surface-only resume.

### Experimento 3 — Acoplamento entre domínios

Usar o mesmo princípio de observador recorrente em pelo menos dois domínios com adaptadores distintos.

Primeira fase sugerida:

1. sistema dinâmico sintético/controlado;
2. sinal temporal não clínico ou EEG sintético/público.

Aplicações posteriores podem incluir áudio, robótica, sensores ambientais, EEG e séries ordenadas por outra variável, como profundidade.

## Estado atual da evidência

Já há implementação pública/reproduzível para:

- estado recorrente explícito;
- channels e route modulation;
- snapshot/capsule;
- full-state restore;
- surface-only control;
- ablações de canais;
- acoplamento temporal;
- experimentos históricos de atratores e continuidade.

A primeira auditoria mostrou que o classificador histórico de `accumulating_fixed_point` era amplo demais para sustentar sozinho a interpretação moderna: ele classificava como `FIXED_POINT` superfícies com movimento residual considerável. Por isso foi definida uma medida direta separando movimento RMS da superfície e do estado recorrente completo.

Após uma fase exploratória com seeds 94–101, foi registrado **antes da coleta seguinte** um critério confirmatório de cinco condições: superfície RMS `<= 0.001`, estado latente RMS `>= 0.010`, razão latente/superfície `>= 10`, tendência de movimento superficial não crescente e norma latente crescente.

Em 28 seeds inéditas (102–129), o **native v9 apresentou 4/28 candidatos AFP no braço limpo**, com razões de movimento latente/superficial entre aproximadamente **24x e 107x**. Três desses quatro candidatos permaneceram dentro do critério após a perturbação secundária. O native v3/v8 apresentou um único caso compartilhado; v2, os scaffolds públicos de v9 five-channel e Demian v1 não apresentaram candidatos sob o critério congelado.

O resultado, portanto, restringe a hipótese em vez de tratá-la como propriedade universal: o regime aparece de forma esparsa e dependente da arquitetura no v9 canônico sob a definição operacional atual.

Separadamente, resultados recentes no Demian v1 mostram forte separação entre similaridade superficial e diferença interna com continuação futura distinta, mas sua superfície permanece dinamicamente ativa. O v1 é tratado atualmente como evidência de **dependência de trajetória latente**, não como exemplo confirmado de AFP.

A segunda etapa causal também foi executada nos quatro candidatos v9. Como no v9 canônico a superfície é exatamente o canal `fast`, foi possível manter a superfície inicial **idêntica** e alterar somente `slow/control`.

O controle de cópia completa permaneceu com gap futuro zero. Ao remover `slow+control`, o gap RMS médio da superfície futura nos candidatos foi aproximadamente **0.102**; ao remover apenas `slow`, aproximadamente **0.097**; ao remover apenas `control`, aproximadamente **0.0034**. Substituir `slow/control` por estados reais da mesma trajetória de 16–96 passos antes também produziu divergência futura mensurável, mesmo mantendo a superfície atual exatamente igual. Aplicar a mesma perturbação externa aos dois braços praticamente não alterou esse efeito.

Isso localiza grande parte da continuidade causal do v9 no canal `slow` e permite separar duas propriedades: a relevância causal do hidden state é geral no v9, enquanto AFP é o subconjunto em que essa relevância persiste simultaneamente com uma superfície operacionalmente quiescente e estado latente acumulando.

## Resultados esperados

O projeto não pressupõe que AFP seja exclusivo do Demian ou que esteja presente em todas as versões.

Entregáveis:

1. protocolo reproduzível e pré-definido para AFP;
2. mapa longitudinal da dinâmica superfície/latente na linhagem;
3. conjunto de ablações e controles causais;
4. figuras e artefatos de resultados;
5. relatório técnico;
6. código e configurações reproduzíveis;
7. resumo/trabalho para evento científico, se os resultados justificarem;
8. demonstração de acoplamento em outro sistema dinâmico.

## Limites explícitos

O projeto não pretende inferir, a partir desses experimentos:

- consciência;
- agência;
- identidade;
- diagnóstico neurológico;
- biomarcadores clínicos;
- superioridade geral do Demian sobre arquiteturas convencionais.

A questão é estritamente sobre **dinâmica recorrente, observabilidade, estado latente, continuidade e consequência causal**.

## O que buscamos no Centec

### Entrada prática — Centec Labs / CVT Fortaleza

- espaço de desenvolvimento e demonstração;
- inserção em ambiente de inovação;
- contato com equipe técnica;
- possibilidade de apresentar protótipos e resultados;
- acompanhamento para transformar o projeto independente em uma linha tecnicamente estruturada.

O projeto é majoritariamente software e não depende de infraestrutura especial para a primeira fase.

### Objetivo institucional — CEP&T

Submeter formalmente a linha ao **Programa de submissão de projetos de P&D e C&T do CEP&T**. Pelas regras públicas atuais, o próprio proponente pode indicar-se como coordenador. A submissão exige plano de trabalho e carta de apresentação assinados eletronicamente via Gov.BR, além do formulário e arquivos de apoio.

Se o projeto for deferido, o Centec informa proponente/coordenador por e-mail e inicia os procedimentos de implementação junto ao(à) orientador(a) indicado(a) para acompanhar o projeto.

Objetivos práticos:

- obter avaliação técnico-científica formal do desenho experimental;
- iniciar acompanhamento por orientador(a) indicado(a) pelo Centec, caso aprovado;
- enquadrar a pesquisa como P&D/C&T com entregáveis e cronograma;
- concorrer a bolsa quando a modalidade/regras aplicáveis permitirem;
- utilizar resultados para participação em eventos, chamadas e colaborações futuras.

## Primeira demonstração presencial

A demo ideal não é um chatbot.

Ela mostra:

```text
duas trajetórias
      |
surface A ~= surface B
      |
latent A != latent B
      |
mesmo estímulo futuro
      |
future A != future B
```

junto de uma visualização da linhagem:

```text
v2 -> v3 -> v8 -> v9 -> v9-5ch -> v1
 |     |     |     |       |       |
surface velocity
latent velocity
AFP prevalence
perturbation response
capsule continuity
```

## Encaminhamento

Executar duas entradas em paralelo:

1. **Centec Labs / CVT Fortaleza:** solicitar uso do espaço como projeto digital de P&D, com o Demian e a bancada experimental como objeto de desenvolvimento e demonstração.
2. **CEP&T:** submeter o plano de trabalho formal da pesquisa de dinâmica latente, com o protocolo AFP, resultados preliminares/confirmatórios, cronograma e documentação de apoio.

O pedido central ao CEP&T é a **avaliação e implementação institucional da linha de pesquisa com acompanhamento técnico-científico**, mantendo como primeira entrega reproduzível o estudo longitudinal + state surgery do native v9.
