# Natural-Language-Processing-Coursework1 23-24
Year3 semester2 Foundation of Natural Language Processing Coursework1
Language identification and classification

Essay questions mark: 38/57
Code mark: 39/43

Overall 77/100

Code feedback: 

***Question 1.1 (7.5 marks) ***
** Testing model by checking its length and selected probabilities...
Model length: 5579336
|* Passed automarker test: 1.25 marks (out of 1.25)
P(h|t): 0.3116466822608252
|* Passed automarker test: 1.25 marks (out of 1.25)
P(u|q): 0.9853501906482038
|* Passed automarker test: 1.25 marks (out of 1.25)
P(z|q): 1.4536908944718608e-06
|* Passed automarker test: 1.25 marks (out of 1.25)
P(j|<s>): 0.005158178014182952
|* Passed automarker test: 1.25 marks (out of 1.25)
P(</s>|e): 0.348077988641086
|* Passed automarker test: 1.25 marks (out of 1.25)
|*
|* Total for Question 1.1: 7.5 marks
**

***Question 1.2 (7.5 marks) ***
Twitter corpus, top 10 entropies: [(2.4921691054394848, ['and', 'here', 'is', 'proof', 'the']), (2.5390025889056127, ['and', 'bailed', 'he', 'here', 'is', 'man', 'on', 'that', 'the']), (2.5584079236733106, ['is', 'the', 'this', 'weather', 'worst']), (2.568653427817313, ['s', 's', 's', 's', 's', 's', 's', 's', 's', 's']), '...']
|* Passed automarker value test: 4 marks (out of 4)
Twitter corpus, bottom 10 entropies: [(17.523736748003564, ['作品によっては怪人でありながらヒーロー', 'あるいはその逆', 'というシチュエーションも多々ありますが', 'そうした事がやれるのもやはり怪人とヒーローと言うカテゴリが完成しているからだと思うんですよね', 'あれだけのバリエーションがありながららしさを失わないデザインにはまさに感服です']), (17.524868750262904, ['ロンブーの淳さんはスピリチュアルスポット', 'セドナーで瞑想を実践してた', 'これらは偶然ではなく必然的に起こっている', '自然は全て絶好のタイミングで教えてくれている', 'そして今が今年最大の大改革時期だ']), (17.5264931699585, ['実物経済と金融との乖離を際限なく広げる', 'レバレッジが金融で儲けるコツだと', 'まるで正義のように叫ぶ連中が多いけど', 'これほど不健全な金融常識はないと思う', '連中は不健全と知りながら', '他の奴がやるから出し抜かれる前に出し抜くのが道理と言わんばかりに群がる']), (17.527615646393077, ['一応ワンセット揃えてみたんだけど', 'イマイチ効果を感じないのよね', 'それよりはオーラソーマとか', '肉体に直接働きかけるタイプのアプローチの方が効き目を感じ取りやすい', '波動系ならバッチよりはホメオパシーの方がわかりやすい']), '...']
|* Passed automarker value test: 3.5 marks (out of 3.5)
|*
|* Total for Question 1.2: 7.5 marks
**

***Question 1.4 (3 marks) 
p(b|('<s>',)) = [2-gram] 0.046511 # bigram probability of 'b' following '<s>'
p(b|('b',)) = [2-gram] 0.007750 # bigram probability of 'b' following 'b'
backing off for ('b', 'q') # Use a lower-order model to calculate the probability of 'q' following 'b'-> the bigram ('b', 'q') is not found in the training data
p(q|()) = [1-gram] 0.000892 # unigram probability of 'q' occurring in any context
p(q|('b',)) = [2-gram] 0.000092 # bigram probability of 'q' following 'b' after backing off
p(</s>|('q',)) = [2-gram] 0.010636 # bigram probability of '</s>' following 'q'
7.85102054894183 # the entropy of 'bbq'

 95 words
|* Marker comment:
|*
|*
|* Hand-examined free text answer. Max length 26+75 words and awarded: 2.5 marks (out of 3)
|*
|* Total for Question 1.4: 2.5 marks
**

***Question 1.6 (10 marks) 
dev tweets, evaluation score: 1.0
|* Passed automarker value test: 1.0 marks (out of 1)
English tweets with non-ascii, evaluation score: 0.5
|* Passed automarker value test: 0.0 marks (out of 2)
large set of tweets: accuracy, evaluation score: 0.8189333333333333
|* Passed automarker value test: 3.5 marks (out of 5)
large set of tweets: timing, evaluation score: 2500.0
|* Passed automarker value test: 2.0 marks (out of 2)
|*
|* Total for Question 1.6: 6.5 marks
**---

***Question 2.1 (15 marks) 

** Part 2.1.1: Vocabulary (1 marks) ***
vocabulary size: 13521
|* Passed automarker test: 1 marks (out of 1)

** Part 2.1.2: Training method (8 marks) ***
Prior: {'V': 0.47766934282005674, 'N': 0.5223306571799433}
|* Passed automarker test: 1 marks (out of 1)
P(('v', 'rose')|V): 0.006913064743369809
|* Passed automarker test: 1 marks (out of 1)
P(('p', 'of')|V): 0.0012190937826217086
|* Passed automarker test: 1 marks (out of 1)
P(('p', 'of')|N): 0.12333945519178972
|* Passed automarker test: 1 marks (out of 1)
P(('n2', '609')|N): 2.2315401420598457e-06
|* Passed automarker test: 1 marks (out of 1)
P(('n2', '609')|V): 2.6766530157362866e-05
|* Passed automarker test: 1 marks (out of 1)
P(('n1', 'million')|V): 0.004917741586184577
|* Passed automarker test: 1 marks (out of 1)
P(('n1', 'million')|N): 0.004933935254094318
|* Passed automarker test: 1 marks (out of 1)

** Part 2.1.3: Prob classify (5 marks) ***
P(d|[('v', 'took')]): {'V': 0.5886037204341108, 'N': 0.41139627956588926}
|* Passed automarker test: 1 marks (out of 1)
P(d|[('v', 'took'), ('n1', 'advantage')]): {'V': 0.1563326709379453, 'N': 0.8436673290620547}
|* Passed automarker test: 1 marks (out of 1)
P(d|[('p', 'to')]): {'V': 0.8124219037837241, 'N': 0.18757809621627583}
|* Passed automarker test: 1 marks (out of 1)
P(d|[('n1', 'dsfjghdkfjgh'), ('p', 'to')]): {'V': 0.8124219037837241, 'N': 0.18757809621627583}
|* Passed automarker test: 1 marks (out of 1)
P(d|[('v', 'values'), ('n1', 'plan'), ('p', 'at'), ('n2', 'billion')]): {'V': 0.9961437486715612, 'N': 0.0038562513284388163}
|* Passed automarker test: 1 marks (out of 1)

** Part 2.1.4: Classify (1 marks) ***
classification: ['V', 'N']
|* Passed automarker test: 1 marks (out of 1)

** Part 2.1.5: Accuracy (0 marks) ***
Overall check for accuracy: 0.7949987620698192
|* Passed automarker test: 0 marks (out of 0)
|*
|* Total for Question 2.1: 15 marks
**---

|* Automarked total: 36.5 marks
|* Hand-marked total: 2.5 marks (out of 3.0)
|*
|* TOTAL FOR ASSIGNMENT: 36.5+2.5 marks
