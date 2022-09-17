# Computer Science Notebook

Computer Science code and notes, starting from the basics.

Data Structures
=

There are several different types of data structures that are used in Computer Science.

**Arrays** are a list of items, sometimes of a particular type. We can nest arrays, creating an *array of arrays* which is also known as a 2-Dimensional array.

**Linked Lists** are a set of items which are linked together by references which form a chain. Items have data stored in them and then store a reference to the next item in the chain. The first item in a list is known as the head, and the last is known as the tail.

**Hash Tables** are a data structure type which takes an input and maps it to a particular bucket using a hash function. That bucket is a place in memory which stores the data that is mapped to by a particular key.

**Graphs** are a data structure which consist of nodes and edges. Each node either is, or is not, connected to any other node in the graph. This forms a set of relationships between each node. If a graph has $n$ nodes and they *are all connected*, then there are $n*(n-1)/2$ total edges. 

**Trees** are a type of graph that have a node known as the root node.  This node is typically depicted at the top of the tree. Trees have a hierarchical structure, with the root being a parent node of some number of child nodes which can themselves be parent notes of subtrees, or leaf nodes. A leaf node is a node that has no children.

**Queues** are a data structure that implement a FIFO (first-in-first-out) mechanism in which the first data to be entered into the queue is the first data to come out. A queue is open at both ends so we can easily add to the back and remove from the front. An example of a queue in the real world would be a job scheduler with single thread execution, where the exact order in which jobs are scheduled is the exact order in which they are executed.

**Stacks** are structure that implement a FILO (first-in-last-out) mechanism in which the first data to be entered into the stack is the last data to come out. Stacks have push and pop methods, pop will remove the top element from the stack, and push will add to the top of the stack. The stack is therefore only open on one end. An example would be a call stack, which keeps track of the list of functions which call each other in a flow of execution. Function $f1()$ might call $f2()$, which calls $f3()$, and these functions will be added into the stack in that order. When they are popped from the stack, $f3()$ will be the first to go, and then $f2()$ and finally $f1()$. 

**Heaps** a heap is a tree that has values associated with each node. A heap must satisfy the *heap property*, which says that each node's parent must have a value greater than or equal to the value of the node itself. That is, if node $P$ is a parent of $N$, and $P$ and $N$ are nodes in a heap, then the value or key, $V(P)$, of node $P$, must satisfy $V(P) >= V(N)$. This one is sometimes hard to remember, so I keep this in my mind: consider that the heap depicted as a tree is actually just an upside down version of a real heap, because in a heap of soil there is more soil on the bottom than on the top. If we had a data structure that satisfied $V(N) >= V(P)$ then that would be closer to what a real heap looks like, but for the heaps as a data structure, it is reversed. 


Cryptography
=

**RSA Cryptosystem**

Some of the information here are my notes from sources linked [here](http://www.cs.sjsu.edu/~stamp/CS265/SecurityEngineering/chapter5_SE/RSAmath.html) and [here](https://books.google.com.au/books/about/Algorithms.html?id=DJSUCgAAQBAJ&source=kp_book_description&redir_esc=y).

Pick primes $p$ and $q$ and multiply them together to get $N$

$\phi(n)$ is Euler's totient function, which is defined as the size of the set of numbers that are less than $n$ and are co-prime with $n$

Two co-prime numbers $n$ and $x$ by definition satisfy $gcd(n, x) = 1$

Therefore if $p$ is a prime number, then $\phi(p) = p-1$ since all numbers less than $p$ are co-prime with $p$

Euler's totient function is such that $\phi(pq) = \phi(p)\phi(q) = (p-1)(q-1)$ (it is "multiplicative")

In our case, the $N = pq$

Therefore $\phi(N) = (p-1)(q-1)$ 

We say $x$ is the multiplicative inverse of $a$ modulo $N$ if $ax = 1 (mod N)$ 

The multiplicative inverse exists if and only if $a$ and $x$ are relatively prime. That is, the largest number that divides them both is 1.

**Fermat's little theorem**

If $p$ is prime and $p$ does not divide $x$, then $x^{p-1} = 1 (mod p)$ 

**Euler's theorem**

Euler's theorem is a generalization of Fermat's little theorem.

If $x$ is relatively prime to $n$ then $x^{\phi(n)} = 1 (mod n)$ 

So if $n$ is prime, then the result of $x$ raised to the power of the length of the set of numbers less than $n$, namely $n-1$, when divided by $n$, has a remainder of 1. 

**Encrypting and decrypting**

We want $ed = 1 mod(\phi(n)) = 1 \mod((p-1)(q-1))$ 

Pick some $e$ that is relatively prime to $\phi(n)$ 

The secret key is $d$, the multiplicative inverse of $e$ modulo $\phi(n)$ calculated using the extended Euclid algorithm.

Now some message, $x \mod N$, can be encrypted by calculating $x^{e} mod N$, and an encrypted message can be decrypted by calculating raising the ciphertext to the power of $e \mod N$.

From above, we know that we want $ed$ such that $ed = 1 \mod\phi(n)$

By the definition of "modulo", we know that there is some $k$ such that $ed = k\phi(N) + 1$, since there is a leftover of $1$ when dividing by $\phi(N)$. 

If $M$ is our message, then $M^{ed} \mod N$

$= M^{(ed -1) + 1} \mod N$

$ = M*M^{ed-1} \mod N$

$ = M*M^{k\phi(N)} \mod N$

$ = M * 1 \mod N $ by Euler's theorem

$ = M \mod N $ i.e. the message