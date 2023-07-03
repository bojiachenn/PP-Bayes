# PP-Bayes
A Prototype Design on Privacy-Preserving Outsourced Bayesian Network

https://link.springer.com/chapter/10.1007/978-3-031-05491-4_7

此架構的 Bayesain Network 模型為預先訓練好的模型。

1. MO 初始化表格
    1. MO先random出一個s
    2. s0 = PRG(s)
    3. s1 = PRG(s0)
    4. 訂一個 PRG founction: PRG(x)
    5. Attribute = sha256(Attribute)
    6. index = index XOR PRG(s0 || index || attribute)
    7. 寫入 [表]1 的 index、attribute，計算機率 = PRG(s1 || id || attribute) (註 id = index)
    8. 拆分機率: [表]2 機率 = 表.機率 - [表]1機率
    9. [表]2 的 index、attribute = [表]1 的 index、attribute
2. MO把 s、空表格 給CSU，[表]2 给CSP
3. CSU 跟 CSP說有哪些資料（Attribute）
4. CSU、CSP 找到路徑
	  1. path = DFS(src, dest) (註 src = Attribute, dest = Target)
5. 由路徑回推，與 CSU 溝通（向 CSU 要哪項index）完成機率乘法（Round trick）
    1. MO把 [a]1、[b]1、[ab]1 給CSU，[a]2、[b]2、[ab]2 給CSU
    2. 做Secure Multiplication
6. 回傳機率較大的 index
