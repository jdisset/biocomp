Ok so what do we want, really?

General plan:
- predict copy numbers at which we'll see the correct computation of a specific band pass

- be able to input a library of parts.
- Convert a sequestron Network to a GRN

At training, we need to know which sequestron architecture we are using.

So, in the case where we have a very large library of L0 parts where all the
weights values can be achieved in various ways, it makes sense to start from a
general NN with no constraints, pick one solution only on the criteria of
accuracy, and then go NN -> SN -> GRN -> MATCH -> GC (which we will call the
forward compilation pass).

However, here we have a slightly different problem. We have a highly
constrained number of L1 parts (and a few L0), and the number of possible SN
architecture is fairly low.

One solution would be to still do the forward compilation pass: unconstrained
NN -> SN -> GRN. but then, it is very likely that a very large number of SN/GRN
will not result in a doable GC.

# general thoughts

the original idea of being able to compile a NN to a GC is theoretically
doable. The compler algorithm already presented should be able to do just that.
However, I believe the compiler in this form will stay in the realm of theory
for a long while, and perhaps even, I'd say that it should.

The problems I see all stem from the fact that the NN abstraction is fairly ill
suited for the biological substrate. In CS terms, it is a very, *very*, leaky
abstraction, in the sense that we are almost certainly going to be faced with
the harsh reality of the constraints of our actual library of parts and have to
deal with "one off" fixes to obtain suboptimal solutions.

That's one major drawback: this abstraction relies so much on the content of
the library of parts that it requires constant back and forth between the
neuromorphic abstraction and the lower levels, for little benefits and great
disadvantages.

The other major drawback is that the neural abstraction is actually unable to
express cleanly some constructs that would otherwise be very efficients. The
only solution to reach these constructs is to proceed to complex optimizations
that reason at a more suited level of abstraction and take into consideration
the real constraints of the biological engineering world.

The purposed benefits are mostly that we can leverage the explosion in
neuromorphic machine learning sucess and litterature and directly apply that to
synthetic biology. This is an obvious source of excitement - that I share.

I argue that other subfields of machine learning and other abstractions are
much more suited than the neuromorphic ones. I also argue that the neural
networks we would be dealing with (a few neurons, a few dozens at best, a few
hundred in our wildest dreams) are not comparable with the ones that are moving
the field of neuromorphic machine learning forward, and would therefore not
benefit from the recent explosion of the field.



# Discrete vs Continuous weights

When training a NN or a SN, the original plan was to proceed to the
weight-matching step which will approximate the required weight using modifiers
and parts from the library. However, with a library restriced in size, there
will be quite a few weights for which we will only have a few possible values.
We can quantize the continuous weights to the closest matching value we can
attain given the current library. But maybe another set of discrete weights
achievable with this parts would actually give better results. One interesting
solution would be to train the network (either in its NN or SN or even GRN
form) with these constraints already known. 

Approaches in litterature: 
1. maintain a set of continuous "hidden" weights that you use for backprop and
that are being quantized during forward pass. Very intuitive and seems to be working OK.
2. use the elegant "Training Discrete-Valued Neural Networks with
Sign Activations Using Weight Distributions". A bit more annoying to implement.

Let's see what this would imply: *TODO*
1. We generate all the possible SN
	a) start from high level NN description

2. We do a forward pass each time to check that a particular SN is possible

3. If a SN is possible, there are probably many different implementation that
   match... so we'd have so many different constraints on the weights.


   hmmm yeah maybe not that great of an idea then.

# Traning a NN vs training a SN
*TODO*

# Types of Sequestrons

Tc = transcription rate
Tl = translation rate
Dg[p/r] = degradation rate (of protein or rna)

## ERN-based

### DNA + RNA inputs

D --> | R | --> out
      |   |
R --> | P |

out = relu( (X[+].Tc[+] - X[-].Tl[-]) / Dgr)

### DNA inputs

       D --> | R1 | -->
             |    |
D --> R2 --> | P  |

out = relu(X[+].W[+]/Dgr1 - X[-].Ts[-].Tl[-]/(Dgr2\*Dgr1))

I guess we can always use the DNA only version and fix Ts = Dgr2 = 1 when assigning species type


# The library of parts

actally there are 2 types of libraries:

## parts

part [_part_id_, name, sequence, part_type={ERN_cut, ERN, OTHER}

or rather everything is a part (because it has a sequence?), and the rest is influence
influence [part_id\*, edge_type, multiplier]


* copy numbers is not part of the library *, rather it is a fundamental property of a dna node.


-> What is the difference between copy number and W ?

D1 -> | + |
      | S | ->
D2 -> | - |

In this case, (general implementation-agnostic Sequestron), W is the
transcription rate (divided by the deg rate).

Copy number can be seen as a third factor, or simply as X (if we don't have any other way to vary X)

In case of a DNA->RNA bias with copy number N
B = N\*Tc/Degr

In case of a DNA->PRT bias with copy number N
B = (N\*Tc/Degr)\*(Tl/Degp)


## predict copy numbers at which we'll see the correct computation of a specific band pass

Ideally we want the compiler to find the circuit from a NN

well technically we already have the circuit.

So we could just model it and learn its params


## Q
- Is the copy number used as an input variation here? or as a weight variation?
We can't really vary the weights through Tc tuning.
-> Well in a Tx we have all the values and we can "simulate" both any weight and any X.
under the assumption that u scales linearly. u = XWN.
most likely u = f(X,W,N)
WARNING: N IS PER PLASMID* -> it's the
same for every connections departing from the DNA node.


- wait but what's the difference between X, W and N again?
-> well it's whatever we want really but I was thinking:
W should be the "fixed" value -> Tc / degr. Encoded in the Gene Circuit.
X is a percentage activation compared to baseline full expression
N is another constant, the copy number. It's applied per DNA node, can't differentiate per connection.
this would be perfect in a perfect world where that total number u would be equal to XWN.
That's probably not the case. c.f Georg data

In this XP, we will have a fixed W and a fixed X, but we'll be able to vary N. 
So we should be able to get a better idea of the scaling.

let's say the NN should work with W = 0.5. (W = Tc / degr)
X [varying 0 -> 1] \* W  is equivalent to N [varying 0 -> 0.5] / (actual Tc / degr)






# The NN -> GC problem

## Naive approach
Going from NN to GC, we will train a NN on a problem, get a result, and we
won't know if the result is doable until we try to compile it. In the case of
this experiment, it is way more liklely that we wont be able to. 

If we manage to get an architecture that's doable, i.e we can compile, we could decide to
then redo a pass of training on the NN with the constraints corresponding to
the GRN architecture (Going GRN -> SN -> NN constraints). That's basically what
we did manually. But, even in this case where manage to compile the NN, the
result might be far from optimal given the library of parts that we have.

We could train lots of NN with random initialization and redo the same process
(NN -> SN -> GRN -> constrained NN, retrain) over and over but that is dumb for
a few reasons:

- we are relying on randomness in the NN stemming from initialization
  variations and/or learning defficiencies in order to explore new feasible
  genetic architectures. It might be that theres a slightly less optimal
  general solution when learning on an unconstrained NN that will actually
  perform much better in real life because much more amenable to
  implementation.

- we are constraining the construction of networks that can easily be
  represented as the specific NN we have initialized (mostly a critic inherent
  to using NNs). Some activation functions (asymmetrical ones) and computations
  are really not Perceptron-like. 
  Ex: 
  - it seems the "real" activation function for an ERN based sequestron is not of the form F(sum of W*X). 
  

## Compilable-SN training

Much better solution: operate at the SN abstraction level. 
We could explore the possible SN, remove the ones we
couldn't implement, and train. That would work
for simple problems and very tiny networks, as it means about 200 SNs for a 2
input 3 nodes, ~3000 for 2 inputs 4 nodes, ~40000 for 5 nodes. 

- We're not reasoning in neurons anymore, we can have asymetrical activation functions

But: 
- We either need to pick an implementation (not scalable and require basically
  to know what we want already)

or

- Bruteforce the space of all SN: That's very quickly an insane amount of networks to try and
  compile, then train. Worst scalablility ever. 

or

- A good solution would be to use an optimization algorithm (EC?) to try
  various architectures in a "smart way"

- 1 SN can have many different implementations and variations, i.e theres not a
  one to one mapping between SN and GC. 
  *TO CHECK*: do some Sequestrons have "hidden variables" that are dependent on
  the implementation (choice of part) and cannot be expressed as a modification
  of the weights?

- we still have some of the disadvantages of the NN abstraction: we need to
  perform some quantization that we won't know until after training and
  compilation. We have to quantize after the fact. For example, let's say a
  Sequestron's output need to be weighted to a value of 0.55 to provide optimal
  results in most training runs. After traning we then try to compile to a Gene
  Circuit implementation, but it turns out there are only 2 values that this
  weight could have: 0.01 and 1. The post-quantization process will pick 1,
  which is very different. We could then decide to do another pass of traning
  with this constraint, but maybe this will result in some parameter changes
  that introduce some necessary very large post-quantization somewhere else in
  the final network. The reason being that the SN abstraction doesn't capture
  some codependencies between parts of the final GN. Ultimately, the optimal
  solution might be theoretically absolutely out of reach. That is a similar
  problem to the one we had with the NN abstraction.


## Machine Learning on Gene Regulatory Networks

There is one final approach that I believe merits a very serious consideration:
machine learning at the Gene Network level of abstraction. Here, the amount of
possible Networks is dependant on the size of the library. Obviously, it can
quickly go through the roof - although in our case we have so few L1s that it's
actually certainly smaller than the amount of possible SN. There is however
some very valuable and successful litterature on how to "grow" similar Computational Networks,
especially in the domain of NeuroEvolution (c.f. Mikkulainen, Stanley, Banzhaf,
...). 

Here, we'd adopt a pure Computational Graph approach, where each node represents a species and each edge represents a computation.

To work entirely on a pure GRN, we would, I believe, need a time domain network,
mostly because of loops, which poses its own set of challenges (such as the
matching of timescales)

However, we can still have some form of hybrid abstraction between a pure
"free" GRN and a SN: a steady-state GRN. This adds a few constraints:
- the computational graph doesn't contain any loop
- there must be a few others, I'll add them later.


Now we have a computational graph whose edges parameters can be modified in a
known way. Indeed, we store in the library information about tunability. For
example, we know that we have, let's say 3 degrons, and we know how they would
transform a translation edge. We can now add this constraint "weight should be
one of these 4 values" to all the translation edges in the graph (which wouldn't
be known in advance if we were operating at the SN level)

In this mode, the big question is: do we learn on unnamed species then compile?
Or do we actually pick stuff from the library before traning. I think the
latter, since we would any way perform this optimization step of assigning
parts to each computational nodes later, we might as well train when they are
there.


# Remarks

In the end what we want is to Learn from data how to perform the desired
computations. One way to look at the impact of the abstraction we choose is in
terms of computational burden in 2 categories: architecture optimization and
parameter optimization. 

With the NN -> GN approach, we have a very loose constraint on final
architecture and rely heavily on parameter optimization to guide the
computations towards the actual function we want. We then "struggle" to piece
together an architecture that can approximate the weights of a NN (that already
performs an approximation of the desired computation). The advantage is that we
have very nice and easy ways to learn the parameters of a NN. It is essentially
a solved problem (although not that easy for tiny tiny networks).

Operating at the GN (or even SN) level, allows for a much more even split
between architecture and parameter optimization. Optimizing architecture is
arguably, on first approach, harder than optimizing weights. However, in my
opinion, this impression is extremely biased by the relatively recent success
of Deep Learning, and ignores a relatively yet untapped source of litterature
and great research on computational graph growth and learning. It also ignores
some of the reasons why deep NN learning is so successful and why it might not
be applicable to our field. The Neural Net abstraction is a universal
approximator, and is amenable to learning thanks to a relatively simple
algorithm that is easily applicable to modern highly parallel hardware. 

It is arguable that the success of Neural Nets come from the fact that we can
always "make them bigger and deeper" in order to learn more and better. The
hardware is there to allow these network's parameters to be learnt. 

As shown in numerous publications (the one that particularly comes to mind is
"the lottery ticket hypothesis"), the sheer size of the Networks used in modern
DL is what allows them to learn things that could actually be performed by much
much smaller networks. 


The good old Perceptron model of the 60's has suddenly been able to thrive and
shine with the democratization of massively parallel computational hardware and
the availability of obscenes amount of training data. 

Unfortunately, biology will deal much less happily with the luxury of bloating
and innefficiency that NN can afford in DL. The NN abstraction uselessly
constrains the space of Gene Circuits that we produce, for a result that is
certainly suboptimal. 

I believe doing machine learning in SynBio is a challenge that would benefit
from a similar effort to what has been done with Neural Network: reviving a
"forgotten" side of Machine Learning research. Combining the evolution of
computational architectures with online learning of the parameters. This, in my
opinion, could lead to the creation of an absolutely transformative tool. It is
also, incidentaly, the way evolution operates.










