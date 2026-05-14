# 00 — Parallelism strategies for distributed training

## The bakery — one analogy to rule them all

Imagine you run a bakery. You have:

- A **recipe book** — this is the model's weights.
- **Raw ingredients** — the input batch (flour, water, etc.).
- **Bakers** — your GPUs.
- **Tasting notes** — gradients, telling you what to adjust about the recipe.
- **A head chef** — the optimizer, who actually updates the recipe each night.

A single baker can handle a single small bakery. But once your menu becomes the size of a phone book and orders come in by the millions, one baker isn't enough. Every parallelism strategy below is a different way of organizing more bakers around the same job.

Hold this picture in your head. We'll come back to it for every strategy.

---

## What we're actually splitting

Before we look at strategies, know what's eating your GPU memory. Four things live there during training:

1. **Parameters** — the weights themselves, the recipe.
2. **Gradients** — one per parameter, the tasting notes.
3. **Optimizer states** — Adam (the most common optimizer) keeps two extra numbers per weight, like the head chef's private notebooks tracking momentum and trends.
4. **Activations** — intermediate forward outputs, kept around because we'll need them during the backward pass. Like dough at various stages between mixing and baking.

For Llama 3 8B in mixed precision, those four things together come out to about **128 GB**. An NVIDIA H100 GPU has **80 GB**. So either you split the work, or you don't train the model. That's the whole reason this document exists.

---

## 1. Data Parallel (DDP) — the franchise

**The picture:** Open identical bakeries in four cities. Each one has the full recipe book and full kitchen. The only difference is they serve different customers. At the end of the day, all four managers hop on a conference call to compare tasting notes and agree on tomorrow's tweaks. By tomorrow morning, every bakery has the same updated recipe again.

That's DDP. Each GPU holds a complete copy of the model and processes a different slice of the batch. After the backward pass, the gradients are averaged across all GPUs (an "all-reduce"), so every GPU applies the same update and stays in sync.

```
              Global batch
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
   ┌────────┐ ┌────────┐ ┌────────┐
   │ GPU 0  │ │ GPU 1  │ │ GPU 2  │
   │ FULL   │ │ FULL   │ │ FULL   │
   │ MODEL  │ │ MODEL  │ │ MODEL  │
   └────┬───┘ └────┬───┘ └────┬───┘
        └─────────┬┴──────────┘
                  ▼
          all-reduce gradients
```

**The catch:** every bakery still needs to fit the entire recipe book. If the book outgrows a single kitchen, DDP can't help you. So DDP is what you reach for when the model fits comfortably on one GPU and you just want to chew through more data faster. Most LoRA fine-tuning runs and anything under ~3B parameters use DDP because it's simple and the communication cost is low.

---

## 2. Fully Sharded Data Parallel (FSDP) — the torn cookbook

**The picture:** The recipe book is now so heavy that no single baker can lift it. So you tear it into pages and hand one page to each baker. They store only their page. When the kitchen needs to cook *chapter 3*, every baker photocopies their pages of chapter 3 and passes them around so that, briefly, every baker has chapter 3 in full. They cook chapter 3 together, then throw away the photocopies and go back to holding only their original page. Move on to chapter 4. Repeat.

That's FSDP. The model's parameters, gradients, and optimizer states are sliced ("sharded") across all GPUs. Each layer's full weights only exist temporarily, gathered just-in-time for the forward pass and then discarded. This is why FSDP can train models that are far bigger than any single GPU.

```
   ┌────────────┬────────────┬────────────┬────────────┐
   │  GPU 0     │  GPU 1     │  GPU 2     │  GPU 3     │
   │ params 0   │ params 1   │ params 2   │ params 3   │  ← steady state
   │ grads  0   │ grads  1   │ grads  2   │ grads  3   │     each GPU holds
   │ optim  0   │ optim  1   │ optim  2   │ optim  3   │     only its slice
   └────────────┴────────────┴────────────┴────────────┘

   For each layer, just-in-time:
     1. all-gather weights from all ranks    (briefly: full layer everywhere)
     2. forward / backward                   (do the work)
     3. free the gathered weights            (back to sharded)
     4. reduce-scatter gradients             (gradients land on right shard)
```

**The trade-off:** you trade memory for network traffic. FSDP roughly triples the per-step communication compared to DDP because of the constant gather-and-discard dance. That's fine on modern interconnects but it's why FSDP performance is sensitive to your network. The payoff is that you can fit models DDP can't touch.

This is the workhorse for training 7B–70B models today. PyTorch's current implementation is called **FSDP2**, exposed via the `fully_shard` API.

---

## 3. Tensor Parallel (TP) — the math problem split column-wise

**The picture:** A single very big matrix multiplication shows up in your kitchen — say, mixing a huge batch of dough where the recipe matrix is 50,000 by 50,000. No baker can do it alone. So you split the *output* column-by-column: baker A produces columns 1–10 of the result, baker B produces columns 11–20, and so on. They each see the full input but only need their slice of the recipe. Stitch the columns back together at the end and you have your answer.

That's tensor parallelism. Inside a *single layer*, the weight matrix is split across GPUs, each GPU does part of the matrix multiply, and the partial results are combined.

```
              Same input X on every GPU
                       │
       ┌───────────┬───┴───┬───────────┐
       ▼           ▼       ▼           ▼
   ┌──────┐   ┌──────┐  ┌──────┐  ┌──────┐
   │ W[:, │   │ W[:, │  │ W[:, │  │ W[:, │
   │ 0:¼] │   │ ¼:½] │  │ ½:¾] │  │ ¾:1] │
   └──┬───┘   └──┬───┘  └──┬───┘  └──┬───┘
      └──────────┴────┬────┴─────────┘
                      ▼
              combine to produce full output
```

**The catch:** the bakers have to talk to each other *during* every layer's computation, not just at the end of the step. That's a huge amount of fast back-and-forth. Tensor parallelism only works well when the GPUs are physically close — same server, connected by NVLink (NVIDIA's super-fast intra-server interconnect, around 900 GB/s on H100). Try TP across machines over a normal network and it falls apart.

Use TP when the layers themselves are too big to fit on one GPU, or when activation memory is the bottleneck. You almost always combine it with FSDP.

---

## What is Megatron?

You'll see "Megatron-style" tensor parallelism mentioned everywhere, so it's worth knowing where this came from.

**Megatron-LM** is a library and a paper from NVIDIA Research, first published in 2019. The team was trying to train language models bigger than would fit on one GPU and they wanted to do it without resorting to the awkward pipelines of the day. They asked a simple question: *given how a transformer is structured, what's the cleanest way to split it across GPUs?*

Their answer was an elegant observation about transformer math. A transformer block is mostly two big matrix multiplies in a row — first an "expansion" projection that makes the hidden dimension bigger, then a "contraction" projection that brings it back down. If you split the first matrix **column-wise** and the second matrix **row-wise**, the partial products from each GPU sum up naturally. You only need *one* all-reduce at the end of each block, instead of one per matrix.

```
   Input X (replicated)
         │
         ▼
   ┌─────────────────────────────────┐
   │  Column-split W1                │   ← each GPU computes some columns
   │  Y_i = X · W1_i (no comm needed) │
   └─────────────────────────────────┘
         │ (no comm)
         ▼
   ┌─────────────────────────────────┐
   │  Row-split W2                   │   ← each GPU multiplies its rows
   │  Z_i = Y_i · W2_i               │      partial sums
   └─────────────────────────────────┘
         │
         ▼
      all-reduce → final Z
```

That column-then-row sandwich is the **Megatron pattern**, and it's the basis of essentially every tensor-parallel transformer trained since. NVIDIA later expanded the project into **Megatron-Core**, which now serves as the engine inside NVIDIA NeMo, and which AWS HyperPod recipes use under the hood for big-model training. So when you see "Megatron-Core integration" in a HyperPod recipe, this is what's happening: the layers are being split using the Shoeybi-et-al column/row pattern, with one all-reduce per block.

**The takeaway:** "Megatron-style TP" is the standard way to do tensor parallelism in transformers. It's not a separate strategy from TP — it's a specific (very smart) implementation of it.

---

## 4. Pipeline Parallel (PP) — the assembly line

**The picture:** Henry Ford's car factory. Worker 1 attaches the chassis. Worker 2 drops in the engine. Worker 3 hangs the doors. Worker 4 paints. A car flows through all four stations in sequence.

That's pipeline parallelism. The model's layers are split into sequential **stages**, each stage living on a different GPU. Layer outputs (activations) flow forward through the stages; gradients flow backward.

```
   Stage 0      Stage 1      Stage 2      Stage 3
   ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
   │ GPU0 │───►│ GPU1 │───►│ GPU2 │───►│ GPU3 │
   │ L1-8 │    │L9-16 │    │L17-24│    │L25-32│
   └──────┘    └──────┘    └──────┘    └──────┘
        ◄─────── backward gradients ────────
```

**The catch:** the **bubble**. The first three stations sit idle waiting for the very first car to reach the painter. And once the line drains at the end of the day, the painter sits idle as the chassis crew finishes the last car. Those idle slots are wasted GPU time.

The fix is **microbatching**: chop your batch into tiny pieces and feed them through the pipeline back-to-back so all stations stay busy most of the time. The classic schedule is called **1F1B** (one forward, one backward), and modern variants like interleaved 1F1B and zero-bubble schedulers have shrunk the idle time dramatically.

Pipeline parallelism is your friend when you need to scale *across servers*, because it only needs to send activations between adjacent stages. That's a tiny amount of traffic compared to TP, and it tolerates network latency well.

---

## 5. Expert Parallel (EP) — the specialist sushi bar

**The picture:** A sushi restaurant with eight chefs. Each chef specializes in one fish — tuna, salmon, eel, etc. As orders come in, the floor manager looks at each ticket and routes it to the right specialist. Different orders go to different chefs in parallel, and the results come back to the customer's plate.

That's expert parallelism, and it's specific to **Mixture-of-Experts (MoE)** models like Mixtral, DeepSeek-V3, and Qwen-MoE. An MoE layer has many "experts" (usually small feed-forward networks); a router decides which experts each token should visit. Expert parallelism puts each expert on a different GPU.

The communication pattern is **all-to-all**: every GPU has tokens that need to go to every other GPU's expert, then come back. Different from the all-reduces of DDP, FSDP, and TP.

You only think about EP if you're training MoE. Otherwise, ignore it.

---

## 6. Context / Sequence Parallel (CP) — the long book

**The picture:** A 100,000-page reference book. No single reader can keep it all on their desk. So you split chapters across readers — but this is tricky, because what each reader needs to understand often depends on context from earlier chapters. **Ring Attention**, the standard implementation, has the readers pass their notes around in a circle so each one eventually sees the keys/values from every other section.

Context parallelism splits along the **sequence dimension** rather than the batch or weight dimensions. You need it when training very long contexts (128k tokens and up), because attention's memory cost grows quadratically with sequence length and quickly dominates everything else.

This is a recent and rapidly evolving area. AWS HyperPod recently used it to extend context windows for foundation models in production.

---

## How they compose: 3D and 4D parallelism

You don't pick *one* — for big models, you stack them. The rule of thumb that almost always works:

- **TP within a single server** (because TP demands the fast NVLink interconnect)
- **PP across servers** (because PP only sends small activation messages, tolerant of slower networks)
- **FSDP or DDP as the outer dimension** (replicating the entire TP/PP setup and feeding it different data shards)

```
   ACROSS NODES (slower network: EFA or InfiniBand)
   ┌─────────────────────────────────────────────┐
   │  FSDP / DP for outer dim                    │
   │  Pipeline stages between nodes              │
   └─────────────────────────────────────────────┘

   WITHIN A NODE (super fast: NVLink)
   ┌──────────────────────────────┐
   │ Tensor Parallel of 8 GPUs    │
   │ G0 G1 G2 G3 G4 G5 G6 G7      │
   └──────────────────────────────┘
```

A practical sizing guide:

| Model size | Typical recipe |
|---|---|
| under 10B | FSDP alone |
| 10B – 100B | FSDP + TP (TP=8 within a node) |
| over 100B | FSDP + TP + PP, plus CP if context is long |

---

## Memorize this table and you're set

| Strategy | Analogy | What's split | When to use |
|---|---|---|---|
| **DDP** | Franchise of bakeries | the batch | model fits on one GPU |
| **FSDP** | Cookbook torn into pages | batch + model state | model too big for one GPU |
| **TP / Megatron** | Math split column-wise | layer weight matrices | layers too big; needs NVLink |
| **PP** | Assembly line | layers in sequence | scaling across servers |
| **EP** | Specialist sushi chefs | MoE experts | MoE models only |
| **CP** | Long book passed around | the sequence dimension | very long context |

The cookbook, the assembly line, the franchise. If you can picture those three, you can recover everything else.
