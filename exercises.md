[Link](https://jax-ml.github.io/scaling-book/)

##### Chapter 1. Rooflines

*Arithmetic intensity*: FLOPs per byte of a kernel or algorithm. One can calculate an accelerator-specific measure Intensity(Accelerator) by dividing its peak FLOPs/s by its peak memory bandwidth; for example, TPU v5e MXU can perform 1.97e14 FLOPs/s and can load 8.2e11 bytes per second from HBM, so its Intensity(Accelerator) is about 240 FLOPs per byte. We can also do the same for communication bandwidth.  For TPU v5e MXU, it sounds like this is 1.97e14 FLOPs/s (still) and 4.5e10 bytes per second between two TPUs. So the same calculation gives 4378 FLOPs per byte. In broad strokes, we hope to implement kernels so that the algorithm's arithmetic intensity is greater than the accelerator-specific measure; that gives us a decent shot of being compute-bound.

Question 1.
* We load a $B\times D$ tensor and a $D\times F$ tensor from memory. INT8 is 1 byte per parameter, so in total we load $D\times (B+F)$ bytes. Similarly, we store $B\times F$ bytes back afterwards.
* It's still (basically) $2\times B\times D\times F$ OPs.
* So the arithmetic intensity is $\dfrac{2BDF}{BD + DF + BF}$.
* $T_{math}$ is $\dfrac{2BDF}{3.94e14}$, and $T_{comms}$ is $\dfrac{BD+DF+BF}{8.1e11}$. A reasonable lower bound is $\max(T_{math}, T_{comms})$ and a reasonable upper-bound is $T_{math}+T_{comms}$.

Question 2.
The number of FLOPs is still $2BDF$ but the amount of memory traffic is now $2BD+DF+2BF$. So the arithmetic intensity is $\dfrac{2BDF}{2BD+DF+2BF}$. 

We saw previously that the Intensity(Accelerator) here is achieved at about 240 FLOPs per byte. If $B<<D$ and $B<<F$, then we can round the denominator to $DF$ and we'll roughly become compute-bound at $2B=240 \implies B=120$.

Question 3.
*iPad plots*

Question 4.
This is a weird one. The number of FLOPs remains $2BDF$ but the amount of memory traffic becomes $BD + BDF + BF$. This means the arithmetic intensity is $\dfrac{2BDF}{BD+BDF+BF}=\dfrac{2DF}{D+DF+F}$, no more dependence on $B$ at all and upper-bounded by $2$.

Question 5.
I can do this one in my sleep. The H100 has roughly $1e15$ BF16 FLOPs and roughly $3.35e12$ bytes per second of memory bandwidth. So its arithmetic intensity is about 300 FLOPs per byte. The arithmetic intensity calculation from before is the same, but now we want $B$ to be at least 300.

##### Chapter 2. TPUs

TPUs are pretty simple compared to GPUs. It's a big compute core called a TensorCore hooked up to a pool of HBM. Each TensorCore has a few big gemm units (MXU), a vectorwise compute unit (VPU), and some SRAM memory for storing tensors (VMEM). It also has a Scalar Unit which seems to basically schedule or program the MXU & VPU? As well as a small amount of SRAM for the Scalar Unit called SMEM.
![[tpu-chip.png]]
Some details:
* The MXU performs a BF16[8, 128] @ BF16[128, 128] -> f32[8, 128] gemm every 8 cycles. At 1.5GHz this translates to about 5e13 gemm FLOPs (calculation: $8\times 128\times 128 \times 2 = 262144$ FLOPs every 8 cycles, and $1.5e9$ cycles per second means $1.5e9\times 262144/8=4.915e13$ flops.)
* VMEM is 128MiB. Seems like a much higher ratio of SRAM per FLOP compared to e.g. the H100, although the guide still makes a lot of comments about VMEM being so small. They say VMEM has about 22x the bandwidth as HBM for v5e which would put it at 1.78e13 bytes per second, about 16 TBps. 
* TPUs can have differing numbers of cores, depending on the need. For example, inference TPUs like v5e only have one core. But most TPUs have two cores which share the same pool of HBM. This is probably a way of becoming compute-bound during inference faster, since decoding is obviously pretty HBM-bandwidth-hungry. 
* TPUs are connected to host through a PCIE gen4 link which runs at about 16 GBps.
* Most TPUs (older generations + v5e + v6e) are kind of connected in a 2D grid-like structure where the links wrap around; they call it a 2D torus. TPU v4 and v5p are connected in a 3D toroidal structure instead. The groupings are called pods, and these pods can get quite big; for example, the largest pods for v5p are 16x20x28 in size. v5e and Trillium are both 16x16 grids; to scale beyond that you need standard DCN. It's quite interesting that Trillium doesn't scale beyond 16x16. I wonder if that's because they leaned into pipeline or expert parallelism more?
* A TPU v5p has 90 GBps of ICI bandwidth "per axis". I think that just means per-link?

Question 1.
A 200B parameter model is 400B=4e11 bytes in BF16. Each TPU v4p has 1.2e12 bytes per second of HBM BW. We can compute: $\dfrac{4e11}{32\times 1.2e12}\approx 10.4$ ms. Interesting that they say it's close to being achievable for small batch sizes.

Question 2.
* Number of hosts:
	* For v5e, There's one host per $4\times 2=8$ TPUs. Since there are $16\times 16=256$ v5e TPUs in a full pod, that's 32 hosts. 
	* For v5p, there's one host per $2\times2\times1=4$ TPUs, and there are $16\times20\times28=8960$ TPUs in a full pod, so there are a whopping 2240 hosts.
* HBM capacity:
	* Each v5e has 16GB of HBM, so there is $256\times16=4096$GB of HBM in a full v5e pod
	* Here it's $96\times8960=860160$GB in a full v5p pod
* BF16 FLOPs:
	* v5e is $256\times 1.97e14=5.04e16$ FLOPs
	* v5p is $8960\times 4.59e14=4.11e18$ FLOPs

Question 3.
The number of flops is still $2BDF$. The data traffic is still $2BD + 2DF + 2BF$ bytes. Assuming $B$ is still much smaller than $D$ and $F=4D$, we still end up with an arithmetic intensity proportional to $B$.

A TPU v6e has 9.2e14 FLOPs. PCIE runs at $1.5e10$ bytes per second, so we're aiming to achieve a ratio higher than $\dfrac{9.2e14}{1.5e10}=6.13e4$. That means we want $B$ on the order of 61000. Maybe that assumption that $B$ is much smaller than $D$ isn't always true at that point...

Question 4.
1. The total number of FLOPs is $B\times 4096 \times 16384\times 2=1.34e8 \times B$ . If we could run at 100% FLOPs utilization, the TPU v5e has $3.94e14$ INT8 FLOPs, so we would run in $\dfrac{B}{2.94e6}$ seconds.
   The total amount of data traffic is $4096 \times 16384 + 4096 \times B + 16384 \times B = 6.71e7 + 20480\times B$ bytes. At 100% bandwidth utilization, this would take $\dfrac{6.71e7 + 20480B}{8.1e11}$ seconds.
   If we can fully overlap compute and memory traffic, our runtime would be $\max\left(\dfrac{B}{2.94e6}, \dfrac{6.71e7 + 20480B}{8.1e11}\right)$ seconds. Setting these equal and solving gives $B\approx 263$ as the breakeven batch size when compute starts to dominate.
2. We can do the same calculation except the memory traffic latency is 22 times smaller: $\dfrac{6.71e7 + 20480B}{22\times 8.1e11}$. The runtime is $\max\left(\dfrac{B}{2.94e6}, \dfrac{6.71e7 + 20480B}{22\times 8.1e11}\right)$ seconds, and the breakeven point is roughly at $B \approx 11$. 

Question 5.
1. There are no wraparound links (otherwise we'd be just two hops away). So instead we need to take six hops to get from `TPU{0, 0}` to `TPU{3, 3}`. Assuming the first byte takes effectively no time per hop beyond the 1$\mu s$ latency, the first byte will arrive in about $6\mu s$.
2. The total size of this tensor is $8\times 128\times 8192 \times 2 = 2^{24}$ bytes. The one-way TPU v5e ICI BW per link is $4.5e10$ bytes per second. ~~So the amount of time to send the data across a single link is $2^{24}/4.5e10=0.373 ms$~~. For each node, we can actually send traffic both vertically and horizontally at the same time -- so `TPU{0, 0}` can get rid of its data with $2\times 4.5e10 = 9e10$ bytes per second, and the total time to do so would be $2^{24}/9e10 = 0.186$ms. If we wanted to get cute, there would also be about $6\mu s$ of pipeline warmup latency, so in total it would look more like $0.192$ms.1. The total number of FLOPs is $B\times 4096 \times 16384\times 2=1.34e8 \times B$ . If we could run at 100% FLOPs utilization, the TPU v5e has $3.94e14$ INT8 FLOPs, so we would run in $\dfrac{B}{2.94e6}$ seconds.
   The total amount of data traffic is $4096 \times 16384 + 4096 \times B + 16384 \times B = 6.71e7 + 20480\times B$ bytes. At 100% bandwidth utilization, this would take $\dfrac{6.71e7 + 20480B}{8.1e11}$ seconds.
   If we can fully overlap compute and memory traffic, our runtime would be $\max\left(\dfrac{B}{2.94e6}, \dfrac{6.71e7 + 20480B}{8.1e11}\right)$ seconds. Setting these equal and solving gives $B\approx 263$ as the breakeven batch size when compute starts to dominate.
2. We can do the same calculation except the memory traffic latency is 22 times smaller: $\dfrac{6.71e7 + 20480B}{22\times 8.1e11}$. The runtime is $\max\left(\dfrac{B}{2.94e6}, \dfrac{6.71e7 + 20480B}{22\times 8.1e11}\right)$ seconds, and the breakeven point is roughly at $B \approx 11$. 

Question 5.
1. There are no wraparound links (otherwise we'd be just two hops away). So instead we need to take six hops to get from `TPU{0, 0}` to `TPU{3, 3}`. Assuming the first byte takes effectively no time per hop beyond the 1$\mu s$ latency, the first byte will arrive in about $6\mu s$.
2. The total size of this tensor is $8\times 128\times 8192 \times 2 = 2^{24}$ bytes. The one-way TPU v5e ICI BW per link is $4.5e10$ bytes per second. ~~So the amount of time to send the data across a single link is $2^{24}/4.5e10=0.373 ms$~~. For each node, we can actually send traffic both vertically and horizontally at the same time -- so `TPU{0, 0}` can get rid of its data with $2\times 4.5e10 = 9e10$ bytes per second, and the total time to do so would be $2^{24}/9e10 = 0.186$ms. If we wanted to get cute, there would also be about $6\mu s$ of pipeline warmup latency, so in total it would look more like $0.192$ms.

Question 6.
* Compute: The gemm takes $8\times (128\times 1024) \times (128\times 1024)\times 2 = 2.749e11$ FLOPs. We've access to $1.97e14$ BF16 FLOPs on TPU{0,0}, so the compute will take 1.4e-3 seconds.
* Memory bandwidth: TPU{0, 0} needs to load the full $(128\times 1024)\times (128\times 1024)$ INT8 tensor from memory, as well as the $8\times (128\times 1024)$ BF16 vectors and store the same-sized tensor back into memory. This is $128^2\times 1024^2 + 8\times 128\times 1024 \times 2 \times 2 = 1.72e10$ bytes. At 8.1e11 bytes per second, this will take 2.12e-2 seconds.
* Host bandwidth: each TPU needs to load a $(128\times 1024 / 4) \times (128 \times 1024 / 4)$ INT8 tensor from host DRAM over PCIE. This is 1.07e9 bytes, which can be loaded at 1.5e10 bytes per second in 7.13e-2 seconds.
* Communication bandwidth: the tensor shards can take various different pathways but in this topology, all shards will need to pass through either TPU{1,0} or TPU{0,1}. In the best-case the traffic is evenly-split between them, and the bottlenecking-link will handle half of all tensor traffic. We know from the memory bandwidth calculation that each TPU holds 1.07e9 bytes; there are 15 such shards we need to copy over, so the traffic that needs to pass through the bottlenecking link is $\dfrac{15\times 1.07e9}{2} = 8.02e9$ bytes. The bandwidth of the bottlenecking link is 4.5e10 bytes per second so this traffic will take 1.78e-1 seconds to go through. 

##### Chapter 3. Sharding

Pop quiz: We're dividing the 128 dim into 16 pieces (2 for $X$, 8 for $Y$), and replicating along the $Z$-axis. So each device holds a $(128 / 16) \times 2048 = 8 \times 2048$ tensor which has size $16384$ bytes. In total, we use $32\times 16384 = 524288$ bytes.

Pop quiz 2: TPU v5e has 9e10 bytes/s of bi-directional BW per link. Each chip has 512 x 8192 x 2 bytes and we're all-gathering just along the $Y$ dim. Each hop takes $\dfrac{2\cdot 512 \cdot 8192 \cdot 2}{9e10}\approx 1.86$e-4 seconds, and we have to go two hops in one direction to finish the all-gather, so in total it will be $\dfrac{ \cdot 2\cdot 512 \cdot 8192 \cdot 2}{9e10}\approx 3.73$e-4 seconds. We could also have plugged in $\dfrac{2048\cdot 8192\cdot 2}{9e10}$ into the formula $\dfrac{V}{W_{ICI}}$ to get the same answer.

If $E=256, F=256$, then a single hop takes $\dfrac{2\cdot 64\cdot 256 \cdot 2}{9e10} \approx 7.3e$-7 seconds based on bandwidth, which means we might be entering into the latency-based regime. Since a single hop takes minimum 1e-6 seconds, we might estimate the all-gather as taking 2e-6 seconds.

After reading the answer: the above is mostly correct, but we require three hops to perform the all-gather since we don't have wraparound links along a size-4 mesh dimension. So the first question's answer is roughly $3\cdot 1.86\approx 5.58e$-6 seconds while the second question's answer is $3e$-6 seconds.

General comments: 
* The all-to-all animation is really cool.
* It's crazy to read all this stuff written out explicitly after rederiving it internally.
* They note that the all-to-all is 4 times cheaper than an all-gather on TPUs. Is that calculation the same for a fully-connected 8-way topology like DGX H100 or SN40L? Example calculation:
	* Suppose we have an array $A[I_X, J]$ that is initially sharded along $I$. Then each chip has $|I|/8$ x $|J|$. If we want to do an all-to-all and reshard on $J$ instead.... we can imagine each chip actually has eight $|I|/8 \times |J|/8$ shards, and what should happen is each chip should send one of these $|I|/8 \times |J|/8$ shards along one of the seven links to the other chips. That means the amount of bi-directional traffic on a link is $2\times |I|/8 \times |J|/8$, as compared to $2\times |I|/8 \times |J|$ for an all-gather. So the all-to-all is actually **eight times more efficient** than an all-gather on a DGX H100.

Question 1: We're basically duplicating it $|Y|\cdot |Z|=16$ times. If it were sharded along all mesh dimension then the answer would be $1$.

Question 2. I think $4\times 4\times 4$ v4p should have the wraparound links. So basically each chip has $256 \times 1024 \times 2$ bytes. If we're all-gathering only on the $X$ dim, then we send $256 \times 1024$ along the bidirectional links twice. The total latency will be $2\cdot \dfrac{256\cdot 1024 \cdot 2}{4.5e10}\approx 2.33$e-5 seconds. If we're all-gathering in both directions, then  So basically each chip has $256 \times 1024 \times 2$ bytes. If we're all-gathering only on the $X$ dim, then we send $256 \times 1024$ along the bidirectional links twice. The total latency will be $2\cdot \dfrac{256\cdot 1024 \cdot 2}{4.5e10}\approx 2.33$e-5 seconds. If we're all-gathering in both directions, then the per-hop latency is the same but we need to make four total hops (two in each direction), so it should take $\approx$ 4.66e-5 seconds. 

For the AllReduce question, we (that is, I) first need to translate the notation. $[B_X, D_Y]\{U_Z\}$ means that we've sharded $B$ on $X$ and $D$ on $Y$, and need to all-reduce along $Z$. That means each shard is still $256 \times 1024$ on each chip, and we could do this by e.g. viewing each $256 \times 1024$ as four $256 \times 256$ shards which we scatter-reduce and all-gather after. This means each hop will be four times faster than the all-gather hops from the first part of this question, but we do twice as many of them. Overall it should be $1.16$e-5 seconds. Note that each hop will take $\dfrac{256\cdot 256\cdot 2}{4.5e10}\approx 2.91e$-6 seconds, so they don't end up latency-bound.

Question 3. We're sending a size $32$ bf16 tensor along links with unidirectional bandwidth of $4.5e10$ bytes per second. Based on bandwidth, this would take $1.4$ ns, so we're well into the latency-dominated regime. Since there are wraparound links for this topology, we need two hops to finish the all-gather, and the latency is $2$ us.

Question 4. 
I will assume we have wraparound links.

Case 2, Strategy 1.
* First we all-gather $Y[D_X, F]$ along $X$. This requires $\dfrac{D\times F\times 2}{W_{ICI}}$ total time per the general formula for bidirectional all-gathers.
* Next we do the full gemm on each chip. This requires $B\times D\times F\times 2$ FLOPs.

Case 4, Strategy 2.
* First we do the sharded gemm on each chip. This takes $B\times (D/|X|) \times F \times 2$ FLOPs.
* Next we have a $B\times F$ tensor on each chip which needs to be all-reduced along the $|X|$ axis. The latency cost is equivalent to twice the time it takes to all-gather a $B_X\times F$ array along $X$, which is $2\times \dfrac{B\times F \times 2}{W_{ICI}}$.

There is always a FLOPs advantage to Strategy 2. If $B<D/2$ (and we're not in the latency-bound regime), then the communication cost is also lesser for Strategy 2, and that is probably the better strategy. If $B > D/2$, this will depend on parameters of the accelerator setup: how much compute and ICI bandwidth we have; the size of $X$; etc.

Question 5: 
First let's assume we don't duplicate any FLOPs across chips. If so, sharding along $B$ and $F$ will require all-gathers while sharding along $D$ will require all-reduces (twice as expensive). So we should only shard along $B$ and $F$. In that case:
* FLOPs per chip: $\dfrac{B\times D\times F \times 2}{4\times 4 \times 4}$. TPU v5p has 4.59e14 FLOPs per chip, which we can put in the denominator for a FLOPs latency.
* Communication latency: All-gathering across three axes takes $\dfrac{B\times F \times 2}{3\times 1.8e11}$. 
We can compute the crossover point as when $\dfrac{D}{4\times 4 \times 4 \times 4.59e14}=\dfrac{1}{3\times 1.8e11}$, which is roughly when $D\approx 54400$. This is pretty large, so you might wonder whether it might actually be worthwhile to partially replicate across one or more axes. In that case though, we lose some our ICI bandwidth, so it doesn't speed things up... unless we fully replicate. That is only worthwhile when $D<850$, which is pretty uncommon, but it's worth noting.

Question 6.
Part 1. We need to do an all-reduce on the $Y$ mesh dim. The size of the tensor in bytes is $I\times K\times 2/4$ and we actually don't have wraparound links in this case. The time it takes to send a $I\times K \times 2 / (4\times 4)$ three hops is $\dfrac{3\times I\times K}{8\times 4.5e10}$. Doing it twice (for the scatter-reduce and the all-gather) will take $\dfrac{3\times I \times K}{4\times 4.5e10}$ seconds. We'll spend $\dfrac{I\times J\times K\times 2}{4\times 4 \times 1.97e14}$ doing compute at 100% FLOPs util.

Part 2. The simplest thing to do is to all-gather $J$ along the $Y$ dim. Similar to above, this will be the time it takes to send $J\times K \times 2 / (4\times 4)$ three hops, so $\dfrac{3\times J\times K}{8\times 4.5e10}$.
After that, we will spend the same amount of time as above doing compute: $\dfrac{I\times J\times K\times 2}{4\times 4 \times 1.97e14}$.

Part 3. We don't need to do any communication in this case. We still spend $\dfrac{I\times J\times K\times 2}{4\times 4 \times 1.97e14}$ doing compute.

Question 7. $B$ and $C$ are both $8192 \times 32768 \times 2= 512MB$. So we can't afford to replicate them on all chips, or even to replicate one and shard the other. If we shard $B$ and $C$ both, the weight matrices will take $128MB$ per chip, leaving $172MB$ for $x$, the output, and intermediate tensors. The unsharded input $x$ has size $128\times 8192\times 2 = 2MB$; the output will be the same; and the intermediate output will only be $8MB$. Actually not that big a deal. We would face a lot more memory pressure if we had to handle many tokens at once, e.g. in prefill.

Assuming we shard $B$ and $C$ both fully across the chips, compute will take $128\times 8192\times 32768\times 2\times 2/4$ total FLOPs, and $\dfrac{128\times 8192\times 32768\times 2\times 2}{4\times 1.97e14}\approx 174$us.  One way to handle the sharding would be to shard along $F$ in both cases; this would necessitate an all-reduce at the end. This will take $2\times\dfrac{2\times 128\times 8192\times 2}{4\times 4.5e10}\approx 46.6$ us. There are ways to reduce the amount of communication latency needed (e.g. funky shardings where we switch sharding dims after $B\cdot x$) but it's not necessary since we're compute-bound.

Question 8.
```
import jax
import jax.numpy as jnp

# Check the number of available devices (should be 8 TPUs)
devices = jax.devices()
print(f"Number of devices: {len(devices)}")

if len(devices) != 8:
    raise ValueError("This example requires exactly 8 devices.")

# Define the global array shape
global_gather_shape = (6144, 8192)
global_gather_array = jnp.arange(jnp.prod(jnp.array(global_gather_shape))).reshape(global_gather_shape)
print(global_gather_array[:8,:8])

global_reduce_shape = (6144*8, 8192)
global_reduce_array = jnp.arange(jnp.prod(jnp.array(global_reduce_shape))).reshape(global_reduce_shape)
print(global_reduce_array[:8,:8])

# Define the sharding specification. We want to shard along the second axis
# across all 8 devices.
mesh_devices = devices  # Use all available devices
mesh = jax.sharding.Mesh(mesh_devices, ('devices',))
shardings = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('devices', None))

# Allocate the sharded array
sharded_gather_array = jax.device_put_sharded(jax.numpy.split(global_gather_array, 8, axis=0), devices)
sharded_reduce_array = jax.device_put_sharded(jax.numpy.split(global_reduce_array, 8, axis=0), devices)
print("shape of the sharded array:", sharded_gather_array.shape)

print("Shape of the sharded array (list of local arrays):", [x.shape for x in sharded_gather_array])

# Now, let's perform an all_gather operation along the sharded dimension (axis 1)

def all_gather_fn(local_array):
    return jax.lax.all_gather(local_array, axis_name='i', axis=0, tiled=True)

def all_reduce_fn(local_array):
    return jax.lax.psum(local_array, axis_name='i')

def scatter_reduce_fn(local_array):
    return jax.lax.psum_scatter(local_array, axis_name='i', scatter_dimension=0, tiled=True)

def all_to_all_fn(local_array):
    return jax.lax.all_to_all(local_array, axis_name='j', split_axis=1, concat_axis=0, tiled=True)

# pmap the all_gather function across the devices
with jax.profiler.trace("/tmp/tensorboard"):
  all_gathered_result = jax.pmap(all_gather_fn, axis_name='i')(sharded_gather_array)
  all_reduced_result = jax.pmap(all_reduce_fn, axis_name='i')(sharded_reduce_array)
  scatter_reduced_result = jax.pmap(scatter_reduce_fn, axis_name='i')(sharded_reduce_array)
  all_to_all_result = jax.pmap(all_to_all_fn, axis_name='j')(sharded_gather_array)

print("Shape of the all-gathered result (list of global arrays):", [x.shape for x in all_gathered_result])
print("Shape of the all-reduced result (list of global arrays):", [x.shape for x in all_reduced_result])
print("Shape of the scatter-reduced result (list of global arrays):", [x.shape for x in scatter_reduced_result])
print("Shape of the all-to-all result (list of global arrays):", [x.shape for x in all_to_all_result])
print(all_gathered_result[0][:8,:8])
print(all_reduced_result[0][:8,:8])
print(scatter_reduced_result[0][:8,:8])
print(all_to_all_result[0][:8,:8])
```
Results: all-gather took 4.27ms, the reduce-scatter took 220us, the all-reduce took 4.45ms, and the all-to-all took 413 us. That's not what I was expecting to be honest...

Question 9. This seems pretty much the same as Question 4 above?

Question 10.
1. If data can only move around in a ring, like TPU0 -> TPU1 -> ... -> TPUD -> TPU0, then each chip's' sharded tensor will need to travel along $D-1$ links. So the total amount of traffic will be $(D-1)\times (N/D) \times N$ scalars.
2. Let's zoom in on the traffic for TPU0's original sharded tensor. Initially we have an $N/D \times N$ tensor and are switching the sharding over. So $N/D \times N/D$ is already on the right chip, and we need to send $N/D \times (D-1) N/D$ to TPU1. After that, another $N/D \times N/D$ is in the right spot, so $N/D \times (D-2)N/D$ needs to be sent to TPU2, etc. Overall, the amount of traffic from this tensor is $N/D \times (1 + 2 + ... + D-1) \times N/D = N/D \times \dfrac{(D-1)D N}{2D} = N/D \times \dfrac{ D-1 N}{2}$.
3. The all-to-all requires half as much traffic as the all-gather and reduce-scatter. The general reason is that we're sending less traffic with each hop with the all-to-all, whereas with the all-gather we're sending $N/D \times N$ each time.
4. Now we can broadcast the $N/D \times N$ tensor in both directions. So the total amount of traffic stays the same, but we effectively have twice as much bandwidth, so it will take half as much time.
5. Now I think we can broadcast $N/D \times (D-1)N/2D$ in each direction for the first hop; then $N/D \times (D-3) N/ 2D$; and so on. The total amount of traffic going in each direction is $N/D \times ((D-1) + (D-3) + \dots + ) \cdot \dfrac{N}{2D}$ . Assuming $D$ is even, this is the sum of the first $D/2$ odd integers, which sums to $D^2/4$. So the traffic in one direction is $N/D \times \dfrac{ND^2}{8D} = \dfrac{N^2}{8}$. This is roughly 4x faster than the unidirectional case.
6. This is again roughly 4x faster.
##### Chapter 4. Transformers

This is mostly review as well, that's nice. A couple of interesting notes:
* Recomputation strategies. One is to only save the input to each layer, called *Block remat*. Another is to save the output of the large GEMMs, maybe QKVO + FFNs?
Worked problems time.

Question 1.
* Parameters:
	* $64 \times (4096 \times 4096 \times 4 + 3\times 4096 \times 4096 \times 4) + 2 \times 4096 \times 32000\approx 17.4B$ parameters.
	* There are $64 \times 4096 \times 4096 \times 4 \approx 4.3B$ attention parameters, so about 25% of them are attention parameters.
	* Each token's KV caches are $2\times 64 \times 4096 \times 1$. $2$ for K & V, 64 for the number of layers, 4096 for the K & V dims, and 1 for 1 byte per INT8 parameter.

Question 2.
Total FLOPs is $B\times D\times F \times 2$, but since we've not sharded on $Z$ there is an extra factor of $Z$ in there: $B\times D \times F \times 2 \times Z$. Per-chip, the total number of FLOPs becomes $B\times D\times F \times 2/ (|X|\times |Y|)$, which means $B\times D\times F\times 2 / (4\times 8)$.

Question 3. It should be $I\times J \times K \times L \times M \times N \times O \times 2$.

Question 4. Let's assume $B=1$ and assume data loads are in BF16 and no intermediate stores (e.g. using something like FlashAttention):
* Q: $N \times H \times T \times 2$
* K and V: $2\times K \times H \times S \times 2$
* Store afterwards: $N\times H\times T \times 2$
Total: $4\times N\times H\times T + 4\times K \times H \times S$. 

FLOPs: 
* QK matmul: $N \times T\times H \times S \times 2$.
* Softmax: $N\times T\times S\times c$ for some constant $c$
* PV matmul: $N\times H\times T\times S\times 2$
Total: $4\times N\times T\times H \times S$, dropping the softmax flops since $H$ should be much larger than $c$.

So the arithmetic intensity is $\dfrac{4NTHS}{4NHT+KHS}= \dfrac{NTS}{NT+KS}$. If $T=S$ as in prefill, then this becomes $\dfrac{NT^2}{T(N+K)}=\dfrac{NT}{N+K}$. During decode, $T=1$ and we get $\dfrac{NS}{N+KS}\approx \dfrac{N}{K}=G$. For TPUv5e, arithmetic intensity of the accelerator is about 240. For a model like Llama 3 70B, where $N=64$ and $K=8$ and $G=8$, we will become compute-bound at $T=240\times 72/64=270$ during prefill and never during decode.

Question 5. Roughly, self-attention flops are equal to $4NTHS$ as seen above. We can compute QKVO flops as:
* Q: $N\times H\times D\times T\times 2$
* K & V: $2\times K\times H\times D\times S\times 2$
* O: $N\times H\times D \times T \times 2$
for a total FLOPs count of $4NHDT + 4KHDS$. Setting these equal:
$$4NTHS = 4NHDT + 4KHDS \implies NTS = NDT + KDS$$
If $T=S$ as in prefill, we can simplify further:
$$NT^2=NDT+KDT \implies NT = ND + KD$$
and we get $T = D + \dfrac{KD}{N}$. For Llama 3 70B, where $D=8192, K=8, N=64$, this is achieved when $T=9216$.

Question 6. We might have to recompute some per-vector ops but these FLOPs are negligible. The main gemm we need to recompute is the QK matmul. This will take $2BNTHS$ FLOPs per layer, and $2BNTHSL$ total FLOPs across all layers.

Question 7. We can use the calculation: activated parameters x tokens x 6 to get:
$37e9 \times 14.8e12 \times 6 = 3.285e24$ FLOPs (ignoring self-attention FLOPs). H800 has about 1500 FP8 TFLOPs per second, so $1500e12 \times 3600 \times 2.79e6 = 15.066e24$ total available FLOPs. Then the model flops utilization would be roughly $3.285/15.066\approx 21.8\%$. In reality this is an underestimate due to attention FLOPs, recomputation, etc.

Question 8. Let's say each copy of the dense MLP block has $3\times I \times D$ parameters. We load $3ID$ bytes and do $6ID$ (INT8?) FLOPs per token, so we need on TPU v5e to have about 240 tokens per expert to be compute-bound. If $S$ is the total number of tokens, we expect each expert to get $S\times \dfrac{k}{E}$ tokens, so we want $S = 240E/k$. For Deepseek in particular, we need $240\times 256/8 = 7680$ tokens to become compute-bound.

##### Chapter 1 Appendix

Example: Consider the MoE block of a single decoder in Deepseek V3, treated as a single block. You could imagine a few different ways of computing the output of this MoE block for a 1000 token prompt:
* One way would to proceed token by token. For each token, there are eight routed experts and one shared expert. We load the shared expert weights and keep them on device; then for each token, we load the eight routed experts, do computation, and store the output.
	* Compute: $(8+1) \times 3 \times 7168 \times 2048 \times 1000 \times 2 = 7.927e11$ FLOPs.
	* Communication: 
		* $1000 \times 7168 \times 2 = 1.4336e7$ bytes to load the hidden states
		* $(1 + 8 \times 1000) \times 3 \times 7168 \times 2048 = 3.524e11$ bytes to load the weights
		* $1000 \times 7168 \times 2 = 1.4336e7$ bytes to store the new hidden states
	* Arithmetic intensity: $7.927e11 / 3.524e11 \approx 2.25$ FLOPs per byte
* The above approach has a quite low arithmetic intensity. One big reason is that we're loading the same expert many times, because we're not taking advantage of the fact that the same expert will likely be chosen by many tokens. Maybe you're not good at grouping them together, for... reasons. Then another way you could implement the MoE block would be to go expert-by-expert, compute the output for every token, and then mask & scale according to the gate output afterwards.
	* Compute: $(256 + 1) \times 3 \times 7168 \times 2048 \times 1000 \times 2 = 2.264e13$ FLOPs.
	* Communication:
		* $1000 \times 7168 \times 2 = 1.4336e7$ bytes to load the hidden states
		* $(1 + 256) \times 3 \times 7168 \times 2048 = 1.132e10$ bytes to load the weights
		* $1000 \times 7168 \times 2 = 1.4336e7$ bytes to store the new hidden states
	* Arithmetic intensity: $2.264e13 / 1.135e10 \approx 3085$ FLOPs per byte
* This is obviously a much higher arithmetic intensity. What if you *can* only do the compute you have to, e.g. with an expert-parallel mapping where you send tokens to the right expert?
	* Compute: $(8 + 1) \times 3 \times 7168 \times 2048 \times 1000 \times 2 = 7.927e11$ FLOPs.
	* Communication:
		* $1000 \times 7168 \times 2 = 1.4336e7$ bytes to load the hidden states
		* $(1 + 256) \times 3 \times 7168 \times 2048 = 1.132e10$ bytes to load the weights
		* $1000 \times 7168 \times 2 = 1.4336e7$ bytes to store the new hidden states
	* Arithmetic intensity: $7.927e11 / 1.135e10 \approx 69.84$ FLOPs per byte.
