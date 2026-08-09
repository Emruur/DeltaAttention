# Slide 0

What is long context

# Slide 1

Long context is a use case where we feed an LLM a large input

The most critical and relevant applications are coding agents

- where we understand large codebases and handle long running generations

Following is retrieval augmented generation

- Where we feed tens/hundreds/thousands of docuemnts to an LLM t ground its answer

Following them we have long vide understanding and document analysis as our important use cases

Why do we want to optimize long context inference

## Slide 2

Long context is slow

here we see the latency of LLama3 with increasing context lengths

at 512k tokens we wait 4 minutes just for the first word
    - coding agents often hit 512K on a middle scale project
    - this is a small model -> large models worse

## Slide 3

Whaat causes this slowness

Its the attention module

Transformers are formed by a chan of layers , each layer has an attention and an MLP module

In the attention module each word communicates with all the other words -> quadratic complexity

Here we see the fraction of latency spent on attention computation, as context growths attention dominates the prefill latncy

## Slide 4

How can we speed up attention

Heres the observation most of the attention acceleration works utilize

Only few token pairs have strong interaction nad they are mostly localized around the diagonal
meaning words mostly communicate with their neighbours

We are going to see how sparse attention works use ths to speed up inferencce

## Slide 5


First lets see how attention works

we have the input which is a sequence of embedding vectors. 
n is seq dimension and d is the embedding dim

We project the input to Q K and V matrices using out learned projections matrices shown with W

Now lets trace the top path, multiplying Q and K gives us the n*n attention matrix. This shows which words strongly attend to each other.

than we appy softmax on querie columns to tget the attention weights

multiplying the attention weights with the Value matrix gives us a n*d output, note that its the same shape as our input so we can just add it back to the input with a residual connection

## Slide 6

So which part of this process is problematic

Look at the scores and weights matrices. They have the size n*n which blows up pretty quickly

A second reason is the back to back HBM traffic
    - in each of these points we move the massive scores matrix to the HBM and read it back
    - HBM is the slow GPU memory

this, specifically the memory blowout limited llm context sizes for a long time

## 7

To solve it we have flash attention, how does it work

the general idea is we process the n*n attention matrix in small tiles

To do that we divide Q and K into Blocks(here we have 4 blocks for query kay and value).we then do the following. 
This slide shows processing query block3 but oll other query blocks are processed simultaneously

We load q3 and k1 and have tile .........

This solves the memory blowout issue and makes things faster yet we still compute all the tiles (this is an exact method not an approximation, we still get the identical attn output) so its still very low on long contexts

## 8

Lets see how xtent tackles ths. xatn is the closest ancestor to our method and we share a lot of methodology

Here we have the full tiled attn matrix, we dont want to compute all the tiles of this as we seen the attn mass is localized to a few regions

for each tile we sample these antidiagonals (which are the attnetion scores within the tiles)

then per each tile we sum the attn scores to get tile scores

then we perform tile selection based on the scores and get a binary block selection matrix

we than compute the attention for those selected matrces

## 9

some detailss on xat

after we compute the scores we dont just perform thresholding or top-k selection
heres what we do

- softmax on queries

..

# 10

we provide this taxonomy of shape oriented and block-sparse methods

shape methods try to predict the attention pattern of the head and take advantage of that

we have Minference and Flexprefill
    - minference is the foundational method here and its only on used in production


block-oriented methods try identify non-inportant attention tiles and skip them
 we have xattn and DPXA which is our method

# 11

Core insight of the paper

Adjactent Query and Key vectors have high cosine simlarity

We perform the experimen where we compute the average adjacent cosine similarity at layers
    - we perform two varitions one with original and one with shuffled row order

We see that we have very high average adjacent simlarity which drops a lot when we shuffle the token order

Suggesting that the natural token order causes adjacent tokens to be similar

## 12

Our contributions use the insight we have observed nd utilizes it in three ways

First two are development steps where we essas the feasability of turning the insight into acceleration

The third is our strongest contribution where we get competetive results using the insight in tile importance estimation


## 13

Here we have delta packing forms the basis of all 3 contributions


lets see how delta packing works, here we have K for example

we place the two trackers. anchor and candidate

We check if the two have a significant difference bw using cosine distance

Here we dont so we set 0 in delta and increment candidate but leace the anchor

here we have a significant difference so we set 1 and move our anchor

repeat the process

than we take the selected vectors and pack them into delta K

here we have the parallel variant where we perform the process in parallel for fixed sized chunks

this makes things much faster

# 14

Lets see how we use delta packing as a direct optimization

we perform delta on K and compress both K and V using the delta K. now we have the compressed K and V 

We are going to nsert them in the self atnetion

mind the dimensions as we swap

note that the attention weights are now n*m multiplying it with the compressed m*l we get a n*d output which makes the compressed diemsnion disapper and match the input dimention

# 15

# 16

delta xattention, we perf

# 17

multiply packed K and Q to get the partial scores

# 19

Lets see how these perform, this is our experimental setup

Ruler and longbench are text benchmarks and vide-mme is a video understanding benchmark 
    - we use qwen2-VL model

On the rexr benchmarks we have ruler

- ıts a synthetic retrieval heavy benchmark
- we have 13 tasks and evaluate our method on these tasks on a range of sequence lengths from 4k to 128K

- its a good stress test on attention health

we have a light dev version for production and ablation and a full ruler for headline comparsions

Longbench has real world tasks like coding QA and summarization

and the tasks range from 3k to 24K context lebgth

first two contributions Delta-Kv compression and DletaXattention are ony tested on our light ruler setup

and only DPXA is evaluated on the whole suite

# 20
results of delta kv compression

Lookng at the average delta kv is comparable to full attention 
and it can reach up to 1.35 and 1.82 speedup over. flash attention

However its ultimately dominated by both speed and accuracy by other works

However it shows the potential of delta packing

# 21

here we see whether content adaptiva packing is better than non-adaptive baselines

here we have euclidian instead of cosine - but cosine is just a bit better so the insight is still relevant

we have random and peridic packing

at a mild compression ratio content adaptive is better then others but not a lot, periodic packing is surprisingly good

altough at higher compression ratios the difference becomes obvious where content adaptation is curuical

# 22

here we have the DeltaXat results

Our accuracy presrvation tricks help a lot here and still have good KV reduction. keping the diagonal recover accuracy back and ...

however we trade a lot of accuracy for halving KV loads which wont yeild 2x speedups so we didnt proceed with custom kernels to realize its speedup and left it here.

# 23 

Here are the important

Full ruler and full longbench

ruler
1. average : great
2. 128k amazing

longbench:
best on overall, shines on repo b coding and passr retrieval
hovewer lacks a bit 

# 24 

look at end to end speedup over flash attention

we have speedup over a range of seqlengths and speedup isolated at 128k






