##### 基础

###### 一章 基本算法

～～ OD上总结了二分的区间设置





###### 二章 数据结构

～～ yxc这里有一个数组模拟





###### 三章 搜索与图论



~~
+二进制状态压缩、队列--数组模拟、lambda回溯、树--数组模拟、树--结构体
DFS, BFS
树与图：深搜，广搜
最短路
最小树
拓扑排序
二分图



～～
图论问题：难在问题转化，将算法题抽象为图论问题，然后建图实现算法



***搜：二进制状态压缩*** 

```c++
#include <iostream>
#include <vector>
using namespace std;

// 1. 设置指定位为1
int setBit(int state, int pos) {
    return state | (1 << pos);
}

// 2. 设置指定位为0
int resetBit(int state, int pos) {
    return state & (~(1 << pos));
}

// 3. 检查指定位是否为1
bool checkBit(int state, int pos) {
    return (state & (1 << pos))!= 0;
}

// 4. 统计二进制中1的个数（常用的位操作技巧）
int countOnes(int state) {
    int count = 0;
    while (state) {
        state &= (state - 1);
        count++;
    }
    return count;
}

// 5. 遍历所有的状态组合（假设有n个状态位）
void iterateAllStates(int n) {
    int totalStates = 1 << n;
    for (int i = 0; i < totalStates; ++i) {
        // 这里可以对每个状态i进行相应的处理，比如输出等
        cout << "当前状态（二进制表示）: " << bitset<n>(i) << endl;
    }
}
```



***数组模拟队列***

```c++
int q[N * N];
int hh = 0, tt = 0;
while(hh <= tt)
{
    t = q[hh ++ ];
    bfs_vist(t);
    for(int i = 0; i < n; i ++ ) q[tt ++ ] = new;
}
```



***DFS***：

排列数字：
	dfs：每次完成一个完整的组合

```c++
/** 排列数字
I: 序列(1~n)中选择一个属
序列(1~n)成的集合进行排列
O: 全部可能的排列情况
 * IO
 I:
3
 O:
1 2 3
1 3 2
2 1 3
2 3 1
3 1 2
3 2 1
*/
// dfs：每次组成一个完整的组合
#include <iostream>

using namespace std;

void solve(int n)
{
	const int N = 10;
	int path[N];
    // 搜索类算法：必定需要确定节点搜索状态 -- 二进制状态压缩最合适
	auto dfs = [&](auto &&self, int u, int state)->void{
        // 搜索层满
	    if (u == n)
	    {
	        for (int i = 0; i < n; i ++ ) printf("%d ", path[i]);
	        puts("");
	        return;
	    }
        // dfs搜索
        // 顺序在第u个位置，遍历(1~n-1)
	    for (int i = 0; i < n; i ++ )
            // 状态压缩优化：取代布尔数组
	        if (!(state >> i & 1))
	        {
	            path[u] = i + 1;
                // 状态压缩：+(1 << i) 表示第i位被占用
	            self(self, u + 1, state + (1 << i));
	        }
	};

	dfs(dfs, 0, 0);

}

int main()
{
	int n;
    scanf("%d", &n);

    solve(n);

    return 0;
}
```

八皇后问题：
	想清楚怎么遍历

```c++
/**八皇后问题
I: 序列(1~n)中选一个数m
m个皇后在二维空间中，不同行、不同列、不同正对角线、不同反对角线
O: 全部可能的情况数量
 * IO
I : 8
O : 
8个皇后的可能摆法
*/
// 逐个位置搜索
// O = 2^(n^2)
#include <iostream>

using namespace std;

void solve(int n)
{
	const int N = 10;
	bool row[N], col[N], dg[N * 2], udg[N * 2];
	char g[N][N];

	int i = 0;

	auto dfs = [&](auto &&self,int x, int y, int s){
        // 皇后放满
		if (s > n) return;
        // 空间占满
	    if (y == n) y = 0, x ++ ;
	    if (x == n)
	    {
	        if (s == n)
	        {
	            for (int i = 0; i < n; i ++ )
                {
                    //puts(g[i]);
                    for(int j = 0; j < n; j ++ ) cout << g[i][j];
                    cout << endl;
                }
	            puts("");
	            cout << "————————" << i ++ << endl;
	        }
	        return;
	    }

        // 不放皇后
	    g[x][y] = '.';
	    self(self, x, y + 1, s);
        // 放皇后
	    if (!row[x] && !col[y] && !dg[x + y] && !udg[x - y + n])
	    {
	        row[x] = col[y] = dg[x + y] = udg[x - y + n] = true;
	        g[x][y] = 'Q';
	        self(self, x, y + 1, s + 1);
	        g[x][y] = '.';
	        row[x] = col[y] = dg[x + y] = udg[x - y + n] = false;
	    }
	};
  dfs(dfs, 0, 0, 0);

}

int main()
{
    int n;

    cin >> n;

    solve(n);

    return 0;
}
```

```c++
/**八皇后问题
I: 序列(1~n)中选一个数m
m个皇后在二维空间中，不同行、不同列、不同正对角线、不同反对角线
O: 全部可能的情况数量
 * IO
I : 8
O :
8个皇后的可能摆法
*/

// 每行必定放一个，因此有单调性 -> 逐层往下搜

#include <iostream>

using namespace std;

const int N = 20;

void solve(int n)
{
	char g[N][N];
	bool col[N], dg[N * 2], udg[N * 2];

	for (int i = 0; i < n; i ++ )
        for (int j = 0; j < n; j ++ )
            g[i][j] = '.';

    int i = 1;

	auto dfs = [&](auto &&self, int u)->void{
	    // 行数叠满
		if (u == n)
	    {
	        for (int i = 0; i < n; i ++ )
            {
                for(int j = 0; j < n; j ++ ) cout << g[i][j];
                cout << endl;
            }
	        puts("");
            cout << "————————" << i ++ << endl;
	        return;
	    }
        // 遍历一行的每一个
	    for (int i = 0; i < n; i ++ )
	        if (!col[i] && !dg[u + i] && !udg[n - u + i])
	        {
	            g[u][i] = 'Q';
	            col[i] = dg[u + i] = udg[n - u + i] = true;
	            self(self, u + 1);
	            col[i] = dg[u + i] = udg[n - u + i] = false;
	            g[u][i] = '.';
	        }
    };

    dfs(dfs, 0);

}
int main()
{
    int n;

    cin >> n;

    solve(n);

    return 0;
}
```



***lambda实现：递归，返回值***

```c++
#include <iostream>

using namespace std;


int main()
{
    auto dfs = [](auto &&self, int n){
        // if(n == 0) return 0;
        // 注释掉后：dfs造成死循环 -- use before deduction of auto
        
        int s = n + self(self, n-1);
        return s;
    };
    
    cout << dfs(dfs, 3) << endl;
    return 0;
}
```



***BFS***:

最短路径的特点：层层递进

```c++
/** 走迷宫
I: 地图尺寸，地图可走0，地图障碍物1
O: 左上到右下最短路径
 * IO
 I:
5 5
0 1 0 0 0
0 1 0 1 0
0 0 0 0 0
0 1 1 1 0
0 0 0 1 0
 O:
8
*/
// dfs：每次组成一个完整的组合
#include <cstring>
#include <iostream>
#include <algorithm>
#include <queue>

using namespace std;

typedef pair<int, int> PII;

void solve()
{
	const int N = 110;

	int n, m;
	int g[N][N], d[N][N];

	cin >> n >> m;
    for (int i = 0; i < n; i ++ )
        for (int j = 0; j < m; j ++ )
            cin >> g[i][j];

	memset(d, -1, sizeof d);
    d[0][0] = 0;

    auto bfs = [&](){
	    int dx[4] = {-1, 0, 1, 0}, dy[4] = {0, 1, 0, -1};

	    // path
	    PII pre_path[N][N];
	    // bfs
		PII q[N * N];
	    q[0] = {0, 0};
	    int hh = 0, tt = 0;
	    while (hh <= tt)
	    {
	        // bfs_visit
	        auto t = q[hh ++ ];
	        for (int i = 0; i < 4; i ++ )
	        {
	            int x = t.first + dx[i], y = t.second + dy[i];
	            if (x >= 0 && x < n && y >= 0 && y < m && g[x][y] == 0 && d[x][y] == -1)
	            {
	                d[x][y] = d[t.first][t.second] + 1;
	                pre_path[x][y] = t;
	                q[++ tt]={x, y};
	            }
	        }
	    }
        // 输出路径：从终点到起点
	    int x = n-1, y = m-1;
	    while(x || y){
            cout << x << ' ' << y << endl;
            auto t = pre_path[x][y];
            x = t.first, y = t.second;
	    }
	    return d[n - 1][m - 1];
    };

    cout << bfs() << endl;

}


int bfs()
{

}

int main()
{

    solve();

    return 0;
}
```



八数码问题

```c++
/** 八数码
I: (3 X 3)二维矩阵；
(3 X 3)二维矩阵；x表示空格，每次数字可移入空格，持续移动数字直到到达目标状态；
O: 最小操作次数；
 * IO
 I:
2 3 4 1 5 x 7 6 8
 O:
19
*/
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <queue>

using namespace std;

int bfs(string start)
{
    string end = "12345678x";

    // 最短距离
    unordered_map<string, int> d;
    d[start] = 0;

    // 搜索空间：确定坐标，然后swap
    int dx[4] = {-1, 0, 1, 0}, dy[4] = {0, 1, 0, -1};
    
    // bfs策略
    queue<string> q;
    q.push(start);
    while (q.size())
    {
        auto t = q.front();
        q.pop();

        if (t == end) return d[t];

        int distance = d[t];

        // 一维坐标转化为二维坐标
        int k = t.find('x');
        int x = k / 3, y = k % 3;

        // 搜索空间
        for (int i = 0; i < 4; i ++ )
        {
            int a = x + dx[i], b = y + dy[i];
            // 搜索空间一定要检查边界
            if (a >= 0 && a < 3 && b >= 0 && b < 3)
            {
                // 状态转移
                swap(t[a * 3 + b], t[k]);
                if (!d.count(t))
                {
                    // 状态转移处理： t已经发生变化
                    d[t] = distance + 1;
                    q.push(t);
                }
                swap(t[a * 3 + b], t[k]);
            }
        }
    }

    return -1;
}

int main()
{
    // 读入技巧
    char s[2];

    string start;
    for (int i = 0; i < 9; i ++ )
    {
        cin >> s;
        start += *s;
    }

    //cout << bfs(start) << endl;

    return 0;
}
```



***树与图的深度优先遍历*** 

树图数据结构实现：问题简化
	树是无环连通图
	无向图是特殊有向图



图存储结构
	邻接矩阵：稠密边(组合概念)
	邻接表：稀疏边(发送映射概念)
		邻接表实现：
			idx轴上，ne表示下一个idx链，e表示节点，h[a]表示第一个链
			h[ e[] ] = ne[] = idx，e[]与ne[]概念不同
			// 树存储结构： a,b -> h[] -> idx -> e[], ne[]
			int n, h[N], e[N * 2], ne[N * 2], idx;
			// 树插入：add：表头插入方式：idx -> e,ne -> h
			**(a,b : 0~n)** e[idx] = b, **(idx)** ne[idx] = h[a], h[a] = idx; 
			idx ++ ;



树的重心：
问题：
	重心(一个节点，删掉该节点后剩余联通部分中最大节点数最小)
解决：
	节点连接的每个部分(上层，子树)
	每部分连通数(上层节点数，子树节点数)
	子树节点数(深度优先遍历搜索节点数)

```c++
/** 树的重心
I: 树节点数；树的存储边；
重心：一个节点，要求其连接的每个联通部分的最大点数为最小
O: 重心对应最大联通图的点数
 * IO
 I:
9
1 2
1 7
1 4
2 8
2 5
4 3
3 9
4 6
 O:
4
*/
// dfs：每次搜索子树的节点总数
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 100010;

// 树存储结构： a,b -> h[] -> idx -> e[], ne[]
int n, h[N], e[N * 2], ne[N * 2], idx=0;
int ans = N;

void add(int a, int b)
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

// 搜索问题必有状态属性
bool st[N];
int dfs(int u)
{
    // dfs_visit
    st[u] = true;

    // dfs 
    int size = 0, sum = 0;
    // 每条边都是从h=-1哪里插入的，边表的末端就是-1
    for (int i = h[u]; i != -1; i = ne[i])
    {
        int j = e[i];
        
        if ( ! st[j])
        {
            self(y, x);
            ans = std::max(ans, dp[x] + dp[y] + 2);
            dp[x] = std::max(dp[x], v + dp[y]);
        }
    }
    // 子树最大联通节点，重心节点数
    size = max(size, n - sum - 1);
    ans = min(ans, size);

    return sum + 1;
}

int main()
{
    scanf("%d", &n);

    memset(h, -1, sizeof h);

    for (int i = 0; i < n - 1; i ++ )
    {
        int a, b;
        scanf("%d%d", &a, &b);
        add(a, b), add(b, a);
    }

    dfs(1);

    printf("%d\n", ans);

    return 0;
}
```



***树与图的广度优先遍历***

图中点的层次
	1号到n号的最短距离 -- 限定条件边权为1

```c++
/** 图中点的层次
I: n, m； 有向边m对；
(n点,m边)有向图，可重边和自环；边长1，点的编号(1~n)；
O: 点1到n的最短距离；
 * IO
 I:
4 5
1 2
2 3
3 4
1 3
1 4
 O:
1
*/
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <queue>

using namespace std;

const int N = 100010;

// 数组模拟有向图(n, m)
int n, m, h[N], e[N], ne[N], idx;
void add(int a, int b)
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

int bfs()
{
    // 最短路问题：distance
    int d[N];
    memset(d, -1, sizeof d);
    d[1] = 0;

    // 数组模拟队列
    int q[N];
    q[0] = 1;
    int hh = 0, tt = 0;
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; i != -1; i = ne[i])
        {
            int j = e[i];
            // 若节点未被访问(d[j] == -1)
            if (d[j] == -1)
            {
                // 更新阶段距离 bfs_vist
                d[j] = d[t] + 1;
                q[ ++ tt] = j;
            }
        }
    }

    // 
    return d[n];
}

int main()
{
    scanf("%d%d", &n, &m);
    memset(h, -1, sizeof h);

    for (int i = 0; i < m; i ++ )
    {
        int a, b;
        scanf("%d%d", &a, &b);
        add(a, b);
    }

    cout << bfs() << endl;

    return 0;
}
```



***拓扑排序*** 

拓扑的存在：(定性的把握)
	无环必定可拓扑，有环必定不可拓扑
拓扑的确定：(定量的把握)
	入度出度
拓扑排序不唯一

```c++
/** 有向图的拓扑排序
I: n, m； 有向边m对；
(n点,m边)有向图，可重边和自环；拓扑排序存在则输出拓扑排序，否则输出-1；
O: 图拓扑排序；
 * IO
 I:
3 3
1 2
2 3
1 3
 O:
1 2 3
*/
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 100010;

// 数组模拟有向图
int n, m;
int h[N], e[N], ne[N], idx;

void add(int a, int b)
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

// 拓扑排序：度策略
int du[N];
// 数组模拟队列
int q[N];
bool topsort()
{
    // bfs策略
    // 数组模拟队列
    int hh = 0, tt = -1;
    for (int i = 1; i <= n; i ++ )
        if (!du[i])
            q[ ++ tt] = i;
    while (hh <= tt)
    {
        int t = q[hh ++ ];

        for (int i = h[t]; i != -1; i = ne[i])
        {
            int j = e[i];
            // 删掉有向边，就是度减一
            if (-- du[j] == 0)
                // 队列存储：顺便保留了拓扑路径
                q[ ++ tt] = j;
        }
    }
    
    // 输出是否存在拓扑排序
    return tt == n - 1;
}

int main()
{
    // 输入图参数
    scanf("%d%d", &n, &m);
    // 头指针idx表初始化
    memset(h, -1, sizeof h);
    // e[]表示节点值，ne[]表示指针idx
    for (int i = 0; i < m; i ++ )
    {
        int a, b;
        scanf("%d%d", &a, &b);
        add(a, b);

        du[b] ++ ;
    }

    if (!topsort()) puts("-1");
    else
    {
        for (int i = 0; i < n; i ++ ) printf("%d ", q[i]);
        puts("");
    }

    return 0;
}
```



***最短路问题*** ：5个算法

简化的最短路：边权为一 -- bfs



最短路问题小总结：
单源：(源头 = 起点)
	正权
		O(n^2) = dijkstra朴素
		O(mlogn) = dijkstra堆优化(偏序)
	含负权
		O(nm) = Bellman-Ford
		O(m~nm) = SPFA
多源：
	O(n^3) = Floyd



稀疏图和稠密图：
m = n 稀疏
m = n^2 稠密



朴素Dijkstra
	O(n^2)：数组不能超10 ^5

```c++
/** Dijkstra求最短路径I
I: (n, m)有向图；可能重边、自环，边权值为正；
图论经典问题；
O: 1到n的最短路径；
 * IO
 I:
3 3
1 2 2
2 3 1
1 3 4

 O:
3

*/

// 素朴Dijkstra
#include <cstring>
#include<iostream>
#include<algorithm>

using namespace std;

const int N = 510;

int n, m;
// 稠密图：邻接矩阵
int g[N][N];

int dijkstra()
{
    // // Dijkstra算法: 源点，距离，全集合，更新
    // 源点
    int goal = 1;
    // 距离
    int dist[N];
    memset(dist, 0x3f, sizeof dist);
    dist[ goal ] = 0;
    // 使用bool数组区分：已寻路、未寻路
    bool st[N] = {};
    // 更新节点
    int i = n; while(i --)
    {
        // 确保每次完整循环
        int t = -1;
        for(int j = 1; j <= n; j ++)
        {
            // t表示每次选取：未访问、距离最短的节点
            if(!st[j] && ( t==-1 || dist[t] > dist[j])) t = j;
        }
        st[t] = true;
        // 更新节点
        for(int j = 1; j <= n; j ++) 
            dist[j] = min(dist[j], dist[t] + g[t][j]);
    }
    // 输出Dijkstra
    if(dist[n] == 0x3f3f3f3f) return -1;
    return dist[n];
}

int main()
{
    scanf("%d%d", &n, &m);
    memset(g, 0x3f, sizeof g);
    // 重边保留最小值
    while(m --)
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        g[a][b] = min(g[a][b], c);
    }

    int t = dijkstra();
    printf("%d\n", t);
    return(0);
}
```



堆优化Dijkstra
	稠密图可以重边
	最短路遍历每条边
	m=n~n^2
	idx是轴，idx对应e，idx-ne连接之前的边指针

```c++
/** Dijkstra求最短路II
I: (n, m)有向图；可能重边、自环；边权值为正；
图论经典问题；
O: 1到n的最短路径，不存在则输出-1；
 * IO
 I:
3 3
1 2 2
2 3 1
1 3 4

 O:
3

*/

// 素朴Dijkstra
#include <cstring>
#include <iostream>
#include <algorithm>
#include <queue>

using namespace std;

typedef pair<int, int> PII;

const int N = 1e6 + 10;

// 邻接表存储有向图：重边不影响最短路，无需插入时保留最小值
int n, m;
int h[N], w[N], e[N], ne[N], idx;
void add(int a, int b, int c)
{
    // idx自己是轴，idx-e对节点一一对应，idx-ne对应轴前面的idx，h初始化-1
    e[idx] = b, ne[idx] = h[a], h[a] = idx;
    w[idx] = c;
    idx ++ ;
}

int dist[N];
int dijkstra()
{
    // dijkstra： 源，距，全，更新
    int goal = 1;
    
    // ！有些编译器中：大数组充当局部变量会导致运行崩溃
    // int dist[N];
    memset(dist, 0x3f, sizeof dist);
    dist[goal] = 0;
    
    // ! 全局转局部变量要小心：初始化
    bool find_set[N] = {};
    
    // 小根堆写法：每次提取权值最小的节点
    // 小根堆实现：STL方式：O(m)=O(n~n^2)
    // 模拟小根堆：能减小维护堆的复杂度O(n)
    // second节点编号，first节点到源点距离
    priority_queue<PII, vector<PII>, greater<PII>> heap;
    heap.push({0, 1});
    while (heap.size())
    {
        auto t = heap.top();
        heap.pop();
        int node = t.second, distance = t.first;
        
        if (! find_set[node]) 
        {
            find_set[node] = true;
            // 遍历节点每个边：边指针以-1为结尾
            for (int i = h[node]; i != -1; i = ne[i])
            {
                // 选取最小路径：当前点node,通过边idx=i,到达j节点
                int j = e[i];
                if (dist[j] > dist[node] + w[i])
                {
                    dist[j] = dist[node] + w[i];
                    heap.push({dist[j], j});
                }
            }
        }
    }
    // dijkstra输出
    if (dist[n] == 0x3f3f3f3f) return -1;
    return dist[n];
}

int main()
{
    scanf("%d%d", &n, &m);

    memset(h, -1, sizeof h);
    while (m -- )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        add(a, b, c);
    }

    printf("%d\n", dijkstra());

    return 0;
}
```



结构实现有向图2：可以省略add()函数

```c++
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 510, M = 10010;

// 邻接表存储方式2(n, m)
int n, m, k;
struct Edge{
    int a, b, c;
} edges[M];

int main()
{
    scanf("%d%d%d", &n, &m, &k);

    for (int i = 0; i < m; i ++ )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        edges[i] = {a, b, c};
    }
    
    return 0;
}
```



Bellman-Ford：控制边数的最短路只能用Bellman-Ford
	相比于dijkstra：不严格控制节点集合，n轮每轮扫描搜索点的所有边(=所有边)
	可以处理负权边：n次，松弛操作
	可以证明：三角不等式，最后所有路径会收敛到最小
	n轮迭代时更新的路径：经过不超过n条边
	寻找负环：n轮仍更新，n条边对应n+1个点，则重点，重点仍更新必定负权
	相比于SPFA：一般SPFA寻负环，且SPFA算法性能更优，但B-F可以控制边数
	代码：
		back_up[]有效控制迭代轮数，实现有效边权更新
		开放判断(>0x3f3f3f3f / 2)：负权时dist[a]=INF会更新dist[b]=INF

```
// Bellman-Ford
for i = 1 ~ n:
    // 遍历当前节点a的所有边j
	for a, j_b, j_w:
		dist[j_b] = min(dist[j_b], dist[a] + j_w)
```

```c++
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 510, M = 10010;

// 邻接表存储方式2(n, m)
int n, m, k;
struct Edge
{
    int a, b, c;
}edges[M];


int dist[N];
int last[N];

void bellman_ford()
{
    // Bellman_Ford算法最短路：源，距，更新
    int goal = 1;
    
    // int dist[N];
    memset(dist, 0x3f, sizeof dist);
    dist[1] = 0;
    
    for (int i = 0; i < k; i ++ )
    {
        // Bellman_Ford：严格控制迭代轮数
        memcpy(last, dist, sizeof dist);
        for (int j = 0; j < m; j ++ )
        {
            // Bellman_Ford更新公式：类似于宽搜策略
            // 松弛搜索：a->b, b->c, 第一轮(a只能到b, b->c不更新), 第二轮(a->c的路径自动更新)
            auto e = edges[j];
            dist[e.b] = min(dist[e.b], last[e.a] + e.c);
        }
    }
}

int main()
{
    scanf("%d%d%d", &n, &m, &k);

    for (int i = 0; i < m; i ++ )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        edges[i] = {a, b, c};
    }

    bellman_ford();

    if (dist[n] > 0x3f3f3f3f / 2) puts("impossible");
    else printf("%d\n", dist[n]);

    return 0;
}
```



SPFA最短路：此算法时间比较优先，但据说容易被卡(估计是卡空间32)
	—— 被卡成O(nm)则换堆优化dijkstra
	—— 网格形状的图容易卡SPFA
	相对于Bellman-Ford算法：优化dist[e.b] = min(dist[e.b], last[e.a] + e.c);
	—— 队列优化dist[e.a]的选取

```
/*
I:(n,m)有向图
(n,m)有向图，可能重边或自环，边权可能为负(=不能dijkstra);
数据保证无负权回路；
O:最短距离或者“impossible”

I:
3 3
1 2 5
2 3 -3
1 3 4
O:
2
*/

#include <cstring>
#include <iostream>
#include <algorithm>
#include <queue>

using namespace std;

const int N = 100010;

// 数组模拟有向图
int n, m;
int h[N], w[N], e[N], ne[N], idx;
void add(int a, int b, int c)
{
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

int spfa()
{
    // SPFA最短路：源，距，队状态，更新(i->j,i到每个邻接的点j)

    int goal = 1;

    int dist[N];
    memset(dist, 0x3f, sizeof dist);
    dist[goal] = 0;

    queue<int> q;
    q.push(1);
    // 布尔数组：管理入队状态，是否在队
    bool st[N];
    st[1] = true;
    while (q.size())
    {
        int t = q.front();
        q.pop();
        st[t] = false;
        // 遍历当前节点i的每条边：边链表以-1结尾
        for (int i = h[t]; i != -1; i = ne[i])
        {
            int j = e[i];
            if (dist[j] > dist[t] + w[i])
            {
                dist[j] = dist[t] + w[i];
                // 通过当前i点，更新了其末端j点，从而j的末端也可以更新
                if (!st[j])
                {
                    q.push(j);
                    st[j] = true;
                }
            }
        }
    }

    return dist[n];
}

int main()
{
    // 数组模拟有向图
    scanf("%d%d", &n, &m);
    memset(h, -1, sizeof h);
    while (m -- )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        add(a, b, c);
    }

    int t = spfa();

    if (t == 0x3f3f3f3f) puts("impossible");
    else printf("%d\n", t);

    return 0;
}
```



SPFA判断负环：
	cnt[]积累路径点数：cnt[]>=n则n点成环

```
/*
I:(n,m)有向图；
(n,m)有向图，重边和自环，边权可能负(不能dijkstra);
判断负权回路；
O:字符串；

I:
3 3
1 2 -1
2 3 4
3 1 -4
O:
Yes
*/

#include <cstring>
#include <iostream>
#include <algorithm>
#include <queue>

using namespace std;

const int N = 2010, M = 10010;

// 有向图：数组模拟
int n, m;
int h[N], w[M], e[M], ne[M], idx;
void add(int a, int b, int c)
{
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

// cnt[]
int dist[N], cnt[N];
bool st[N];

bool spfa()
{
    // SPFA求负权回路：源，距，全，更新
    
    // 无源点
    
    // 距离dist[N]：无需初始化为INF
    // 0权回路a->b=-1, b->a=1；
    // 负权回路 a->b=-2, b->a=1;
    // 从负边哪里更新，只要负边存在，就一定会每轮都增加负边终点j的路径长度
    
    // 队列集合状态：st[]
    
    // 当前点t到终点j
    // BFS策略
    queue<int> q;
    for (int i = 1; i <= n; i ++ )
    {
        q.push(i);
        st[i] = true;
    }
    while (q.size())
    {
        int t = q.front();
        q.pop();
        st[t] = false;
        for (int i = h[t]; i != -1; i = ne[i])
        {
            int j = e[i];
            if (dist[j] > dist[t] + w[i])
            {
                dist[j] = dist[t] + w[i];
                cnt[j] = cnt[t] + 1;

                if (cnt[j] >= n) return true;
                if (!st[j])
                {
                    q.push(j);
                    st[j] = true;
                }
            }
        }
    }

    return false;
}

int main()
{
    // 头节点指针
    memset(h, -1, sizeof h);
    // 输入有向图
    scanf("%d%d", &n, &m);
    while (m -- )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        add(a, b, c);
    }

    if (spfa()) puts("Yes");
    else puts("No");

    return 0;
}
```



Floyd
	转化数组为最短路数组：BFS的理解方式理解此迭代过程
	—— 理解动态规划，也可以类似的抽象来理解
	—— 抽象的思路，可以抽象节点具像化理解深搜和广搜的概念

```
/*
I:(n,m)有向图;
(n,m)有向图，可能重边、自环，边权可能为负；
k个询问，x,y之间的最短距离，若不存在则输出“impossible”
O:每个询问的最小路径长度，不存在则输出“impossible”;

I:
3 3 2
1 2 1
2 3 2
1 3 1
2 1
1 3
O:
impossible
1
*/

#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 210, INF = 1e9;

int n, m, Q;
int d[N][N];

// floyd算法：将图邻接矩阵g[][]转化为最短距离矩阵d[][]
void floyd()
{
    // floyd算法需要邻接矩阵特殊初始化：有权(自己到自己是默认有权)，或者INF
    // 邻接矩阵d[][]不是到原点的距离：只是点到点的距离

    // 经过k:1~n次迭代，更新路径
    // BFS思路可以理解此算法的有效性：初始还不能被更新到，迭代轮数达到后就自然而然更新到
    for (int k = 1; k <= n; k ++ )
        for (int i = 1; i <= n; i ++ )
            for (int j = 1; j <= n; j ++ )
                d[i][j] = min(d[i][j], d[i][k] + d[k][j]);
}

int main()
{
    scanf("%d%d%d", &n, &m, &Q);

    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= n; j ++ )
            if (i == j) d[i][j] = 0;
            else d[i][j] = INF;
    while (m -- )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        d[a][b] = min(d[a][b], c);
    }

    floyd();

    while (Q -- )
    {
        int a, b;
        scanf("%d%d", &a, &b);

        int t = d[a][b];
        if (t > INF / 2) puts("impossible");
        else printf("%d\n", t);
    }

    return 0;
}
```





***最小树*** 

～～
Prim：
	朴素O(n^2)
	堆优化O(mlogn)：需要优化时一般不用prim优化版而是直接用Kruskal算法
Kruskal：O(mlogm)



Prim:
	set_dist

```
/*
I:(n,m)无向图；m (u, v, w)；
（n,m）无向图，可能重边、自环，边权可能为负；
O:最小生成树的边权和，不存在最小生成树则输出“impossible”；

I:
4 5
1 2 1
1 3 2
1 4 3
2 3 2
3 4 4
O:
6
*/


#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 510, INF = 0x3f3f3f3f;

// 无向图：邻接矩阵存储
int n, m;
int g[N][N];


int dist[N];
bool min_set[N];
int prim()
{
    // dist[]不是节点到源点的距离，而是：1初始值 2一条边的值代表点到点集的距离
    memset(dist, 0x3f, sizeof dist);
    // n次迭代
    int res = 0;
    for (int i = 0; i < n; i ++ )
    {
        // 每次找到：最小树集合外 + 距离最短(需要dist[t]有一个初始化)
        int t = -1;
        for (int j = 1; j <= n; j ++ )
            if (!min_set[j] && (t == -1 || dist[t] > dist[j]))
                t = j;

        if (i && dist[t] == INF) return INF;
        // 防止自环更新自己的dist[]
        if (i) res += dist[t];
        min_set[t] = true;
        // 使用集合更新距离：一条边的权重
        for (int j = 1; j <= n; j ++ ) dist[j] = min(dist[j], g[t][j]);
    }

    return res;
}


int main()
{
    // 输入无向图：邻接矩阵
    scanf("%d%d", &n, &m);
    memset(g, 0x3f, sizeof g);
    while (m -- )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        // 应对重边、自环min
        // 有向图代替无向图：邻接矩阵为对称阵
        g[a][b] = g[b][a] = min(g[a][b], c);
    }

    int t = prim();

    if (t == INF) puts("impossible");
    else printf("%d\n", t);

    return 0;
}
```



Kruskal
	稀疏图
	思路简单：边权排序，遍历单调边(并查集应用)
	简单数结：存边即可a b w

```
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 100010, M = 200010, INF = 0x3f3f3f3f;

// 只需要存储边
int n, m;
struct Edge
{
    int a, b, w;

    bool operator< (const Edge &W)const
    {
        return w < W.w;
    }
}edges[M];

// 并查集操作：查看是否在一个集合
int p[N];
int find(int x)
{
    // 并查集数组模拟：为根则返回，否则往根回溯
    if (p[x] != x) p[x] = find(p[x]);
    return p[x];
}

// Kruskal输出最小生成树权重和
int kruskal()
{
    // Kruskal: 边排序，从小到大遍历边
    
    // 边排序
    sort(edges, edges + m);

    // 从小到大遍历边：m
    for (int i = 1; i <= n; i ++ ) p[i] = i;    
    int res = 0, cnt = 0;
    for (int i = 0; i < m; i ++ )
    {
        int a = edges[i].a, b = edges[i].b, w = edges[i].w;
        a = find(a), b = find(b);
        // 仅在不连通加入边
        if (a != b)
        {
            p[b] = a;
            res += w;
            // 判断图是否连通 cnt
            cnt ++ ;
        }
    }

    if (cnt < n - 1) return INF;
    return res;
}

int main()
{
    // 输入有向图
    scanf("%d%d", &n, &m);
    for (int i = 0; i < m; i ++ )
    {
        int a, b, w;
        scanf("%d%d%d", &a, &b, &w);
        edges[i] = {a, b, w};
    }

    int t = kruskal();

    if (t == INF) puts("impossible");
    else printf("%d\n", t);

    return 0;
}
```



***二分图***

～～
染色法O(n+m)
匈牙利二分图<=O(nm)



染色法判定二分图
	判断是否为二分图

```
/*
I:n,m, m(u, v)
(n,m)无向图；可能重边、自环；判断是否为二分图；
O:“Yes”/“No”

I:
4 4
1 3
1 4
2 3
2 4

O:
Yes
*/

// 模拟输入：无向图 add(a,b),add(b,a)
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 100010, M = 200010;

// 无向图-数组模拟
int n, m;
int h[N], e[M], ne[M], idx;
void add(int a, int b)
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

// 染色法判断二分图
int color[N];
bool dfs(int u, int c)
{
    // 每个dfs处理一个节点：当前节点u: 染色，遍历各条边
    // 染色设置：1，2，目标节点j不可与起始节点u同色
    color[u] = c;
    for (int i = h[u]; i != -1; i = ne[i])
    {
        int j = e[i];
        if (!color[j])
        {
            if (!dfs(j, 3 - c)) return false;
        }
        else if (color[j] == c) return false;
    }

    return true;
}

int main()
{
    // 图数组模拟：有向边模拟无向边(add(a,b),add(b,a))
    scanf("%d%d", &n, &m);
    memset(h, -1, sizeof h);
    while (m -- )
    {
        int a, b;
        scanf("%d%d", &a, &b);
        add(a, b), add(b, a);
    }
    // 遍历各个节点i以染色
    bool flag = true;
    for (int i = 1; i <= n; i ++ )
        if (!color[i])
        {
            if (!dfs(i, 1))
            {
                flag = false;
                break;
            }
        }

    if (flag) puts("Yes");
    else puts("No");

    return 0;
}
```



二分图的最大匹配

```
/*
I n1, n2, m (u, v)
二分图左n1右n2边m，数据保证边不同部
二分图匹配：两点一边的子图成为一个匹配
二分图最大匹配：图的匹配中边数最大的匹配
O 最大匹配数

I
2 2 4
1 1
1 2
2 1
2 2

O
2

*/

#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 510, M = 100010;

// (二分图=树=近似有向图来看待)数组模拟：邻接表：发送的概念
int n1, n2, m;
int h[N], e[M], ne[M], idx;
void add(int a, int b)
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

int match[N];
bool st[N];
bool find(int x)
{
    // 遍历当前节点x所有连接边对应的节点j
    for (int i = h[x]; i != -1; i = ne[i])
    {
        int j = e[i];
        if (!st[j])
        {
            // 有没有考虑过st
            st[j] = true;
            // 条件：没有匹配或者可以换个匹配
            if (match[j] == 0 || find(match[j]))
            {
                match[j] = x;
                return true;
            }
        }
    }

    return false;
}

int main()
{
    scanf("%d%d%d", &n1, &n2, &m);
    memset(h, -1, sizeof h);
    while (m -- )
    {
        int a, b;
        scanf("%d%d", &a, &b);
        add(a, b);
    }

    int res = 0;
    for (int i = 1; i <= n1; i ++ )
    {
        memset(st, false, sizeof st);
        if (find(i)) res ++ ;
    }

    printf("%d\n", res);

    return 0;
}
```











###### 四章 数学知识

～～
质数3
约数3
欧里几德算法
欧拉函数，与欧拉函数相关的欧拉定理
快速幂2
扩展欧里几德算法
中国剩余定理



～～
数论题一定要算空间复杂度：判断数组大小是否会超



***质数｜素数***
～质数｜素数：只有1和自身两个因数
	i <= n / i ：n中最多只有一个大于根号n的质因数
判断质数
分解质因数
筛质数



判断质数

```
/*
判断质数
	试除法：O(n)
	试除法边界条件：i <= 
	大于sqrt(n)之多只有一个：if(res > 1)
*/

/*
I n ：a
判断每个a是否为质数
O Yes / No

I
2
2
6

O
Yes
No
*/

#include <iostream>
#include <algorithm>

using namespace std;

bool is_prime(int x)
{
    // 数据集合边界
    if (x < 2) return false;
    // i <= x / i; 1）防止乘法溢出 2）防止计算开方
    // 找出的i就是因数
    for (int i = 2; i <= x / i; i ++ )
        if (x % i == 0)
    return false;
    return true;
}

int main()
{
    int n;
    cin >> n;

    while (n -- )
    {
        int x;
        cin >> x;
        if (is_prime(x)) puts("Yes");
        else puts("No");
    }

    return 0;
}
```



分解质因数

```
/*
I n : a
分解质因数a = a1^n1 * ... * an ^ nn，从小到大输出质因数及其指数
O 底数，指数

I
2
6
8

O
2 1
3 1

2 3
*/

#include <iostream>
#include <algorithm>

using namespace std;

void divide(int x)
{
    // 寻找质因数 ： 1 循环条件，2 判断因数条件
    for (int i = 2; i <= x / i; i ++ )
        if (x % i == 0)
        {
            int s = 0;
            while (x % i == 0) x /= i, s ++ ;
            cout << i << ' ' << s << endl;
        }
    if (x > 1) cout << x << ' ' << 1 << endl;
    cout << endl;
}

int main()
{
    int n;
    cin >> n;
    while (n -- )
    {
        int x;
        cin >> x;
        divide(x);
    }

    return 0;
}
```



筛质数
	朴素筛法：O(nlgn = (∑n/i) )
	埃氏筛法：O(n/lgn * lgn = n * lglgn ≈ n) 底数不同lglgn
	线性筛法：10^7会快一倍10^6差不多 -- 每轮都将下一轮的质数筛掉(1, 2, 3, 4, 5, 6, 7)

```
/*
I n 
求1～n中质数个数
O 质数个数

I
8

O
4

*/

#include <iostream>
#include <algorithm>

using namespace std;

const int N= 1000010;

int primes[N] = {}, cnt = 0;
bool is_n_p[N] = {};

void get_primes(int n)
{
    /*
    // 朴素筛法
    for (int i = 2; i <= n; i ++ )
    {
        // 按序cnt存储质数i
        if (! is_n_p[i]) primes[cnt ++ ] = i;
        // lgn：筛选删除序列后面的倍数
        for (int j = i + i; j <= n; j += i)
            is_n_p[j] = true;
    }
    */
    /*
    // 埃氏筛法：l = i初始化，r = n
    // no_prime[N], primes[N] -- 范围无限制，因为：区间有限，区间内素数数量有限
    for (int i = 2; i <= n; i ++ )
    {
        // 按序cnt存储质数i
        if (! is_n_p[i]) 
        {
            primes[cnt ++ ] = i;
            // lgn：筛选删除序列后面的倍数
            for (int j = i + i; j <= n; j += i)
                is_n_p[j] = true;
        }
    }
    */
    // 线性筛法
    // 刚好每一轮i将后一轮非prime筛选掉: 1,2,3,4,5,6,7
    // primes[j]到当前数: primes[j]一定是质数，当前i可能为质数可能为合数
    // 平行逻辑：既消后合数，又推出循环
    for (int i = 2; i <= n; i ++ )
    {
        if (!is_n_p[i]) primes[cnt ++ ] = i;
        for (int j = 0; primes[j] <= n / i; j ++ )
        {
            is_n_p[primes[j] * i] = true;
            if (i % primes[j] == 0) break; 
        }
    }
}

int main()
{
    int n;
    cin >> n;

    get_primes(n);

    cout << cnt << endl;

    return 0;
}
```







***约数｜因数***
～～ 
	分解质因数：N=p1^a1 * p2^a2 * ... * pn^an
	约数：n=p1^b1 * ... * pn^bn
试除法求约数：n
约数个数：(a1 + 1) * (a2 + 1) * ... * (an + 1)
约数之和：(p1^0 + p1^1 + ... + p1^n) * ... * (pn^0 + pn^1 + ... + pn^n)
最大公约数：欧几里得算法：约数d|a, d|b => d|(a * x + b * y)；最大公约数(a, b) = (b, a mod b)；



试除法求约数

```
/*
试除法求约数
	与质因数不同：
		质因数是构成一个数
		约数：可以被整除而已，约数之间不是互质的（2，4）
*/

/*
I n ：a
对于每个a，顺序输出其所有约数
O 所有约数

I
2
6
8

O
1 2 3 6 
1 2 4 8 

*/

/*
试除法求约数
	与质因数不同：
		质因数是构成一个数
		约数：可以被整除而已，约数之间不是互质的（2，4）
*/

#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

vector<int> get_divisors(int x)
{
    // 求出较小的约数，直接得到另一个
    vector<int> res;
    for (int i = 1; i <= x / i; i ++ )
        if (x % i == 0)
        {
            res.push_back(i);
            // 不同添加两个，相同只添加一个
            if (i != x / i) res.push_back(x / i);
        }
    sort(res.begin(), res.end());
    return res;
}

int main()
{
    int n;
    cin >> n;

    while (n -- )
    {
        int x;
        cin >> x;
        auto res = get_divisors(x);

        for (auto x : res) cout << x << ' ';
        cout << endl;
    }

    return 0;
}
```



约数个数

```
/*
约数个数:
  公式计算：求乘（i指数 + 1）
*/

/*
I n(1-100) : a(1-2*10e9)
求a1 * ... * an的约数个数，答案对1e9 + 7取模
O 输出整数

I
3
2
6
8

O
12

*/

#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <vector>

using namespace std;

typedef long long LL;

const int N = 110, mod = 1e9 + 7;

int main()
{
    int n;
    cin >> n;

    unordered_map<int, int> primes;

    while (n -- )
    {
        int x;
        cin >> x;
        // 存储<质因数，次数>primes
        for (int i = 2; i <= x / i; i ++ )
            while (x % i == 0)
            {
                x /= i;
                primes[i] ++ ;
            }
        // 处理最后一个质因数：！！！大于sqrt(n)的质因数只可能有一个
        if (x > 1) primes[x] ++ ;
    }

    LL res = 1;
    for (auto p : primes) res = res * (p.second + 1) % mod;

    cout << res << endl;

    return 0;
}
```



约数之和

```
/*
约数之和：
  公式计算：求乘(i)(求和(j)底数i^j)
	与分解质因数不同（4: 质因数2，约数124），质因数是约数子集
*/
/*
I n ： a
计算a1 * ... * an 对1e9+7取模队结果
O 整数

I
3
2
6
8

O
252

*/

#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <vector>

using namespace std;

typedef long long LL;

const int N = 110, mod = 1e9 + 7;

int main()
{
    int n;
    cin >> n;

    unordered_map<int, int> primes;

    while (n -- )
    {
        int x;
        cin >> x;
        // 分解质因数：得到质因数及其次数
        for (int i = 2; i <= x / i; i ++ )
            while (x % i == 0)
            {
                x /= i;
                primes[i] ++ ;
            }

        if (x > 1) primes[x] ++ ;
    }
    
    // 计算约数之和：p1^0 + ... + p1^n = ((p1 + 1) * ... * p1 + 1)
    LL res = 1;
    for (auto p : primes)
    {
        LL a = p.first, b = p.second;
        LL t = 1;
        while (b -- ) t = (t * a + 1) % mod;
        res = res * t % mod;
    }

    cout << res << endl;

    return 0;
}
```



最大公约数：欧里几德算法

```
/*
欧里几德算法
	(a,b) = (b, a mod b)：证明()
*/

/*
I n ： ai， bi
每对数值（ai， bi）的最大公约数
O 每行输出最大公约数

I
2
3 6
4 6

O
3
2

*/

#include <iostream>
#include <algorithm>

using namespace std;

// 根据公式(a, b) = (b, a mod b)求最大公约数：不断递归
int gcd(int a, int b)
{
    return b ? gcd(b, a % b) : a;
}

int main()
{
    int n;
    cin >> n;
    while (n -- )
    {
        int a, b;
        scanf("%d%d", &a, &b);
        printf("%d\n", gcd(a, b));
    }

    return 0;
}
```



***欧拉函数***
～～
欧拉函数phi(n)：1～n内与n互质数的数量
	应用：欧拉定理(a与n互质 => a^phi(n) 三 1 (mod n) ) -- 同余条件下的剩余系可以证明
筛法求欧拉函数：O(n)求一个序列每个数的欧拉函数



欧拉函数

```
/*
欧拉函数
	1～n内与n互质数的数量
	容斥原理证明公式：phi(N) = N * (1 - 1 / p1) * (1 - 1 / p2) * ... * (1 - 1/pn)
	证明：容斥原理==画集合图不断并集然后减去重合部分==综合法顺序推导公式然后公式合并变为一个大因式
	phi(N)=N-求和(N/pi)+求和(N/pipj)-...+求乘(1/pi)*N
*/

/*
I n, n:(a)
数a的欧拉函数
O a的质数的个数

I
3
3
6
8

O
2
2
4
*/

#include <iostream>

using namespace std;


int phi(int x)
{
    int res = x;
    // 得到序列质数：2～x之间的i即为质数
    // i <= x / i : 1避免乘法溢出 2避免开方计算
    for (int i = 2; i <= x / i; i ++ )
        if (x % i == 0)
        {
            // 欧拉公式换个写法：不溢出且不多省：res * (1 - 1 / i)
            res = res / i * (i - 1);
            // 完全消除质数：i^k，用循环
            while (x % i == 0) x /= i;
        }
    if (x > 1) res = res / x * (x - 1);

    return res;
}

int main()
{
    int n;
    cin >> n;
    while (n -- )
    {
        int x;
        cin >> x;
        cout << phi(x) << endl;
    }

    return 0;
}
```



线性筛法求欧拉函数

```
/*
线性筛法：三种情况(每次遍历得到i),每次都是一轮一轮的到值
	质数：质数的（1～n）互质个数p-1=i-1
	质数倍数：N=求乘(pi^ni), phi(pj * i) = pj * phi(i) = phi(i)*pj
	非质数倍数：N相对于i多一个质因子pj：phi(pj*i)=pj * phi(i) * (1-1/pj)=phi(i)*(pj-1)
*/

/*
I 整数n
计算1～n所有整数的欧拉函数之和=求和(phi(i))
O 欧拉函数之和

I
6

O
12

*/

#include <iostream>

using namespace std;

using LL = long long;

const int N = 1000010;


int primes[N], cnt;
int euler[N];
bool st[N];


void get_eulers(int n)
{
    // 规定
    euler[1] = 1;
    // 线性筛法：每次筛选判断i，i对应的欧拉函数存入euler[]，筛后选值进入primes[]
    // 三种Euler[]赋值：质数，后向筛选(当前i为质因子、不为质因子)
    for (int i = 2; i <= n; i ++ )
    {
        if (!st[i])
        {
            primes[cnt ++ ] = i;
            euler[i] = i - 1;
        }
        for (int j = 0; primes[j] <= n / i; j ++ )
        {
            int t = primes[j] * i;
            st[t] = true;
            if (i % primes[j] == 0)
            {
                euler[t] = euler[i] * primes[j];
                // 停止一轮的筛选：break
                break;
            }
            else euler[t] = euler[i] * (primes[j] - 1);
        }
    }
}


int main()
{
    int n;
    cin >> n;

    get_eulers(n);

    LL res = 0;
    for (int i = 1; i <= n; i ++ ) res += euler[i];

    cout << res << endl;

    return 0;
}
```





***快速幂*** 

～～
快速幂计算：log2(k)
快速幂求逆元

```
/*
分解快速幂运算过程：qmi = a ^ k % p
	a ^ k = a ^ (k的二进制表示=每位为0或1)
	a^(2^1 * 1位) * ... * a^(2^n * n位) = a^(求和2^i*i位)
	求模等于分开求模再组合求模
*/

/*
I n : ai, bi, pi 
计算快速幂
O 快速幂

I 
2
3 2 5
4 3 9

O
4
1
*/

#include <iostream>
#include <algorithm>

using namespace std;

// 数论计算过程中可能会超过int
typedef long long LL;

// 快速幂
LL qmi(int a, int b, int p)
{
    // 初始化
    LL res = 1 % p;
    while (b)
    {
        // 分解幂形式 a ^ (c0 * 2^0 + c1 * 2^1 + ... + cn * 2^n)
        // 每轮更新一位，且从b的小位开始更新
        if (b & 1) res = res * a % p;
        a = a * (LL)a % p;
        b >>= 1;
    }
    return res;
}


int main()
{
    int n;
    scanf("%d", &n);
    while (n -- )
    {
        int a, b, p;
        scanf("%d%d%d", &a, &b, &p);
        printf("%lld\n", qmi(a, b, p));
    }

    return 0;
}
```



快速幂求逆元

```
/*
快速幂求逆元：
    快速幂的逆元==在取模的条件下的逆元
    费马定理(要求p为质数)：1 = b^(p-1) = b * b^(p-2) = b * x（mod p素数）=> b^(p-2)=x
    快速幂的逆元存在性：b与p互质
*/

/*
I n : (ai, pi)
模运算下的逆元，其值的范围为1～p-1
O 存在逆元则输出否则输出“impossible”

I
3
4 3
8 5
6 3

O
1
2
impossible

*/

#include <iostream>
#include <algorithm>

using namespace std;

typedef long long LL;


LL qmi(int a, int b, int p)
{
    LL res = 1;
    while (b)
    {
        if (b & 1) res = res * a % p;
        a = a * (LL)a % p;
        b >>= 1;
    }
    return res;
}


int main()
{
    int n;
    scanf("%d", &n);
    while (n -- )
    {
        int a, p;
        scanf("%d%d", &a, &p);
        if (a % p == 0) puts("impossible");
        // 使用费马定理直接推导快速幂的逆元的公式：b * x = 1(mod p)
        else printf("%lld\n", qmi(a, p - 2, p));
    }

    return 0;
}
```



***扩展欧里几德算法***



扩展欧里几德算法

```
/*
裴蜀定理：公因数构造
	任意a，b，存在非零整数x、y，使得（a，b）= x * a + y * b
扩展欧里几德算法：一种构造x、y的方法，构造的方式证明裴蜀定理
	(a, 0): x = 1, y = 0
	(a, b) -- x, y
	(b, a mod b): d = b * y + (a mod b) * x 
									= b * y + (a - 下整a/b * b) * x 
	                = b * (y - 下整a/b * x) + a * x
*/


/*
I n：ai， bi
构造x，y，使得ai*xi + bi*yi = gcd(ai, bi)
O xi, yi

I
2
4 6
8 18

O
-1 1
-2 1
*/

#include <iostream>
#include <algorithm>

using namespace std;

// 扩展欧里几德算法：构造最大公因数的配比x，y
int exgcd(int a, int b, int &x, int &y)
{
    // 递归基：b == 0
    if (!b)
    {
        x = 1, y = 0;
        return a;
    }
    int d = exgcd(b, a % b, y, x);
    y -= a / b * x;
    return d;
}

int main()
{
    int n;
    scanf("%d", &n);

    while (n -- )
    {
        int a, b;
        scanf("%d%d", &a, &b);
        int x, y;
        exgcd(a, b, x, y);
        printf("%d %d\n", x, y);
    }

    return 0;
}
```



扩欧应用：线性同余方程

```
/*
线性同余方程转化
	a*x 三 b（mod m）=> 存在y，使得a*x = m*y + b <=> a*x + m*(-y) = b 
	扩展欧里几德算法求解同余方程的充分必要条件：必定有a*x + b*y = d, 若d是b的倍数，则上述方程有解
*/

/*
I n：ai， bi， mi 
求解线性同余方程：ai * xi 三 bi （mod mi）
O 有解输出xi，无解输出“impossible”

I
2
2 3 6
4 3 5

O
impossible
-3

*/

#include <iostream>
#include <algorithm>

using namespace std;

typedef long long LL;

// 推导公式：
int exgcd(int a, int b, int &x, int &y)
{
    if (!b)
    {
        x = 1, y = 0;
        return a;
    }
    int d = exgcd(b, a % b, y, x);
    y -= a / b * x;
    return d;
}


int main()
{
    int n;
    scanf("%d", &n);
    while (n -- )
    {
        int a, b, m;
        scanf("%d%d%d", &a, &b, &m);

        int x, y;
        int d = exgcd(a, m, x, y);
        if (b % d) puts("impossible");
        else printf("%d\n", (LL)b / d * x % m);
    }

    return 0;
}
```



***中国剩余定理***



扩展欧里几德算法求逆：因为不一定是质数

证明可以剩余得到适合的剩余值

中国剩余定理：
	存在mi互质，使成立方程组(i) : x mod mi = ai，
	记：M=求乘mi，Mi=M/mi，Mi^(-1)为Mi数论倒数，
	则： 原方程组解为：x=求乘ai * Mi^(-1) * Mi + k * M, (k为整数)，变动k则可得到各种x

```
/*
I：n，n：ai，mi
按照中国剩余定理可以分解x，现在给出n：ai，mi反求x值
O：最小的满足中国剩余定理的x

I
2
8 7
11 9

O
31

*/

#include <iostream>
#include <algorithm>

using namespace std;

typedef long long LL;

// 扩展欧里几德算法
LL exgcd(LL a, LL b, LL &x, LL &y)
{
    if (!b)
    {
        x = 1, y = 0;
        return a;
    }

    LL d = exgcd(b, a % b, y, x);
    y -= a / b * x;
    return d;
}


int main()
{
    int n;
    cin >> n;

    // 应用：扩展欧里几德算法两个一组地求x，y
    // 使用扩展欧里几德算法因为，mi不一定为质数，不一定满足费马定理条件
    LL x = 0, m1, a1;
    cin >> m1 >> a1;
    for (int i = 0; i < n - 1; i ++ )
    {
        LL m2, a2;
        cin >> m2 >> a2;
        LL k1, k2;
        LL d = exgcd(m1, m2, k1, k2);
        // ai有两个相等
        if ((a2 - a1) % d)
        {
            x = -1;
            break;
        }

        k1 *= (a2 - a1) / d;
        k1 = (k1 % (m2/d) + m2/d) % (m2/d);

        x = k1 * m1 + a1;

        LL m = abs(m1 / d * m2);
        a1 = k1 * m1 + a1;
        m1 = m;
    }

    if (x != -1) x = (a1 % m1 + m1) % m1;

    cout << x << endl;

    return 0;
}
```



###### 五章 动态规划 



～～
DP
高维度：可以显示直观的状态转移
优化减少维度：可以优化维度f[N] [N]到f[N]，但是就可能会(上一层转移则需要j=m逆方向，本层则无需)
DP实现
dp通过递归实现：记忆化搜索



～～
01背包：二维，一维优化
完全背包：暴力三重，二重，一维优化
多重背包：暴力枚举，二进制优化
分组背包：



问题：
	01背包：体积-价值
	完全背包：任意数量，一个种类选取一定数量
	多重背包：i物品ni件
	分组背包：n类每类ni件，类别+个数+价值



阶段、决策、最优子结构 --》抽象集合角度理解dp过程
	状态表示：集合表示，属性限制
	状态转移：集合划分
—— {1-i-1}中选择到{1-i}：范围扩大，必定价值总量增加
—— 状态转移方程：f [ i ] [ j ] = max ( f [i - 1] [j] , f [i] [ j - vi ] + wi ) 



实现上：高纬度降为低维度时的更新实现是后往前，与贪心的单调方向相反
	很多dp的实现都是从后往前：就是为了保证一轮轮之间互不干扰 
		结构关系，高向低遍历则不影响从低向高选Max







***01背包问题*** 二维可以优化为一维

```
/*
DP类型：
    状态压缩
    数位
    线性

DP问题
    状态：集合性质按照纬度分析，初始化
    转移：抽象的深搜和广搜的迭代过程

*/

/*
I n, v, n : vi, wi
01背包问题，N件物品，M个空间，第i件物品(vi体积，wi价值)
O 收集物品使得总价值最大

I
4 5
1 2
2 4
3 4
4 5

O
8

*/

#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1010;

// ！本题：价值(体积) -- 最核心对应，因此可以化为一维

// n物品，m体积限制
// 1~N表示物品的轴，v表示体积，w表示价值
// f表示(物品数量，体积)对应的总价值
int n, m;
int v[N], w[N];
int f[N][N];

int main()
{
    cin >> n >> m;

    for (int i = 1; i <= n; i ++ ) cin >> v[i] >> w[i];

    // 单个纬度上直接是贪心策略，多个维度多次迭代以实现全局最优
    // 这个抽象过程类似于弗洛伊德算法求最短路，抽象的深搜与广搜过程
    // 最初始状态：f[0][0]=0，所以第一个初始化时，f[1][0] = 0
    // 类似于搜索过程：第i行，搜索完j列
    // i = 1 对应于取一样物品，观察最大价值组合
    // 然后i1增长就对应着迭代增长，就可以遍历数量i的所有物品组合
    for (int i = 1; i <= n; i ++ )
        for (int j = 0; j <= m; j ++ )
        {
            f[i][j] = f[i - 1][j];
            if (j >= v[i]) f[i][j] = max(f[i][j], f[i - 1][j - v[i]] + w[i]);
        }

    cout << f[n][m] << endl;

    return 0;
}
```

```
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1010;

int n, m;
int v[N], w[N];
int f[N];

int main()
{
    cin >> n >> m;

    for (int i = 1; i <= n; i ++ ) cin >> v[i] >> w[i];
		
		// 01背包的另解：二维简化为一维
		// 1）f[i][j]仅仅用到f[i-1][j]，2）j-vi不大于j
		// 逆序遍历f，使得状态转移成功：
		// j=vi顺序遍历时f[i][j] = max(f[i][j], f[i][j - vi]) ！= max(fij, f[i-1][j-vi]) 没有从i-1到i
		// 所以需要逆序遍历
		// 否则不能一层层地迭代更新：会导致更新混乱
    for (int i = 1; i <= n; i ++ )
        for (int j = m; j >= v[i]; j -- )
            f[j] = max(f[j], f[j - v[i]] + w[i]);

    cout << f[m] << endl;

    return 0;
}
```



***完全背包问题*** 三维可优化为二维

会超时：三重循环会超时
三重循环：最直观地表达状态一层层地转移
二重循环：通过简化表达式，进行表达式替换
二维降一维：不是那么地直观表达状态转移，即会从反方向表达状态转移

```
/*
完全背包：三重循环

状态转移：f[i][j] = max(f[i,j], f[i-1,j-vi*k] + wi*k)
*/

#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1010;

int n, m;
int v[N], w[N];
int f[N][N];

int main()
{
    cin >> n >> m;
    for (int i = 1; i <= n; i ++ ) cin >> v[i] >> w[i];
    
    for (int i = 1; i <= n; i ++ )
        for (int j = 0; j <= m; j ++ )
            for (int k = 0; k * v[i] <= j; k ++ )
            {
                f[i][j] = max(f[i][j], f[i - 1][j - v[i] * k] + w[i] * k);
            }

    cout << f[n][m] << endl;

    return 0;
}
```

```

/*
完全背包：二重循环
*/

#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1010;

int n, m;
int v[N], w[N];
int f[N][N];

int main()
{
    cin >> n >> m;
    for (int i = 1; i <= n; i++) cin >> v[i] >> w[i];

    for (int i = 1; i <= n; i++)
        for (int j = 0; j <= m; j++)
        {
            f[i][j] = f[i-1][j];
            if(j >= v[i])f[i][j] = max(f[i - 1][j], f[i][j - v[i]] + w[i]);
        }

    cout << f[n][m] << endl;

    return 0;
}
```

01背包对比完全背包：最后只差一个循环顺序：第一个参数初始化方式相同，第二个参数有一个k倍参数要找
	01：f[i,j] = Max(f[i - 1],j], f[i - 1,j - v]+w) 
	完全：f[i,j] = Max(f[i - 1,j], f[i,j - v]+w)

```
/*
I n, m. n: vi, wi
完全背包问题：n类物品，每类无限种，背包体积m，求那些物品装入使得总价值最大
O 最大价值

I
4 5
1 2
2 4
3 4
4 5

O
10

*/

/*
完全背包：二重循环 + 一维状态转移
*/

#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1010;

int n, m;
int v[N], w[N];
int f[N];

int main()
{
    cin >> n >> m;
    for (int i = 1; i <= n; i++) cin >> v[i] >> w[i];

    for (int i = 1; i <= n; i++)
        for (int j = v[i]; j <= m; j++)
            f[j] = max(f[j], f[j - v[i]] + w[i]);

    cout << f[m] << endl;

    return 0;
}
```

 

***多重背包问题一：暴力枚举*** 

```
/*
I n,v, n: vi,wi,si
多重背包：i种物品si件，计算可以装下的最大价值
O 最大价值

I
4 5
1 2 3
2 4 1
3 4 3
4 5 2

O
10
*/

#include <iostream>
#include <algorithm>

using namespace std;

const int N = 110;

int n, m;
int v[N], w[N], s[N];
int f[N][N];

int main()
{
    cin >> n >> m;

    for (int i = 1; i <= n; i++) cin >> v[i] >> w[i] >> s[i];

    for (int i = 1; i <= n; i++)
        for (int j = 0; j <= m; j++)
            for (int k = 0; k <= s[i] && k * v[i] <= j; k++)
                f[i][j] = max(f[i][j], f[i - 1][j - v[i] * k] + w[i] * k);

    cout << f[n][m] << endl;
    return 0;
}
```



***多重背包问题二：二进制优化*** 

```

/*
多重背包优化：二进制优化
*/

#include <iostream>
#include <algorithm>

using namespace std;

const int N = 12010, M = 2010;

int n, m;
int v[N], w[N];
int f[M];

int main()
{
    cin >> n >> m;

    int cnt = 0;
    for (int i = 1; i <= n; i ++ )
    {
        int a, b, s;
        cin >> a >> b >> s;
        int k = 1;
        while (k <= s)
        {
            cnt ++ ;
            v[cnt] = a * k;
            w[cnt] = b * k;
            s -= k;
            k *= 2;
        }
        if (s > 0)
        {
            cnt ++ ;
            v[cnt] = a * s;
            w[cnt] = b * s;
        }
    }

    n = cnt;

    for (int i = 1; i <= n; i ++ )
        for (int j = m; j >= v[i]; j -- )
            f[j] = max(f[j], f[j - v[i]] + w[i]);

    cout << f[m] << endl;

    return 0;
}
```



***分组背包问题*** 



```
/*
I n，v，n：si，vij，wij
有n组，总空间v，每组共si件，每组仅可选一件
O 最大价值

I
3 5
2
1 2
2 4
1
3 4
1
4 5

O
8

*/

#include <iostream>
#include <algorithm>

using namespace std;

const int N = 110;

int n, m;
int v[N][N], w[N][N], s[N];
int f[N];

int main()
{
    cin >> n >> m;

    for (int i = 1; i <= n; i ++ )
    {
        cin >> s[i];
        for (int j = 0; j < s[i]; j ++ )
            cin >> v[i][j] >> w[i][j];
    }

    for (int i = 1; i <= n; i ++ )
        for (int j = m; j >= 0; j -- )
            for (int k = 0; k < s[i]; k ++ )
                if (v[i][k] <= j)
                    f[j] = max(f[j], f[j - v[i][k]] + w[i][k]);

    cout << f[m] << endl;

    return 0;
}
```



***数字三角形*** 



```
/*
数字三角形设置：只能那么状态转移
*/

/*
I n, n：第一行1，每行增加1，一共输出n行
输入数字三角形，从顶部出发移动到底层，找出一条路径，使得路径上的数字之和为最大
O 最大路经上数字之和

I
5
7
3 8
8 1 0 
2 7 4 4
4 5 2 6 5

O
30

*/

#include <iostream>
#include <algorithm>

using namespace std;

const int N = 510, INF = 1e9;

int n;
int w[N][N];
int dp[N][N];

int main()
{
    scanf("%d", &n);
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= i; j ++ )
            scanf("%d", &w[i][j]);

    for (int i = 0; i <= n; i ++ )
        for (int j = 0; j <= i + 1; j ++ )
            dp[i][j] = -INF;

    dp[1][1] = w[1][1];
    for (int i = 2; i <= n; i ++ )
        for (int j = 1; j <= i; j ++ )
            dp[i][j] = max(dp[i - 1][j - 1] + w[i][j], dp[i - 1][j] + w[i][j]);

    int res = -INF;
    for (int i = 1; i <= n; i ++ ) res = max(res, dp[n][i]);

    printf("%d\n", res);
    return 0;
}
```



***最长上升子序列*** 



```
/*
n ^ 2做法：最为经典的实现方式
nlgn的做法：优化状态表示dp[i]表示单调递增子序列的末尾元素大小的值
*/

/*
I n，n：ai
给定一个n长度的序列，求序列的严格单调上升自序列
O 最大上升子序列的长度

I
7
3 1 2 1 8 5 6

O
4

*/

#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1010;

int n;
int a[N], dp[N];

int main()
{
    scanf("%d", &n);
    for (int i = 1; i <= n; i ++ ) scanf("%d", &a[i]);

    for (int i = 1; i <= n; i ++ )
    {
        dp[i] = 1; 
        for (int j = 1; j < i; j ++ )
            if (a[j] < a[i])
                dp[i] = max(dp[i], dp[j] + 1);
    }

    int res = 0;
    for (int i = 1; i <= n; i ++ ) res = max(res, dp[i]);

    printf("%d\n", res);

    return 0;
}
```



***最长公共子序列*** 



```
/*
I n， m，s1，s2
最长公共子序列问题
O 最长长度

I
4 5
acbd
abedc

O
3

*/

#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1010;

int n, m;
char a[N], b[N];
int f[N][N];

int main()
{
    scanf("%d%d", &n, &m);
    scanf("%s%s", a + 1, b + 1);

    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= m; j ++ )
        {
            f[i][j] = max(f[i - 1][j], f[i][j - 1]);
            if (a[i] == b[j]) f[i][j] 
            = max(f[i][j], f[i - 1][j - 1] + 1);
        }

    printf("%d\n", f[n][m]);

    return 0;
}
```



区间DP：***石子合并*** 



```
/*
区间DP
*/

/*
石子合并：仅能合并相邻石子
I n，n：每堆石子质量
每次仅能合并相邻石子，合并消耗石子质量之和，求最小消耗
O 最小代价

I
4
1 3 5 2

O
22

*/

#include <iostream>
#include <algorithm>

using namespace std;

const int N = 310;

int n;
int s[N];
int dp[N][N];

int main()
{
    scanf("%d", &n);
    for (int i = 1; i <= n; i ++ ) scanf("%d", &s[i]);

    for (int i = 1; i <= n; i ++ ) s[i] += s[i - 1];

    for (int len = 2; len <= n; len ++ )
        for (int i = 1; i + len - 1 <= n; i ++ )
        {
            int l = i, r = i + len - 1;
            dp[l][r] = 1e8;
            for (int k = l; k < r; k ++ )
                dp[l][r] = min(dp[l][r], dp[l][k] + dp[k + 1][r] + s[r] - s[l - 1]);
        }

    printf("%d\n", dp[1][n]);
    return 0;
}
```



数位统计DP：***计数问题*** 



```
/*
I 不定组数a，b
计数[a, b]内所有数中0～9出现次数
O 每个输入对应每个数字的出现次数

I
1 10
44 497
346 542
1199 1748
1496 1403
1004 503
1714 190
1317 854
1976 494
1001 1960
0 0

O
1 2 1 1 1 1 1 1 1 1
85 185 185 185 190 96 96 96 95 93
40 40 40 93 136 82 40 40 40 40
115 666 215 215 214 205 205 154 105 106
16 113 19 20 114 20 20 19 19 16
107 105 100 101 101 197 200 200 200 200
413 1133 503 503 503 502 502 417 402 412
196 512 186 104 87 93 97 97 142 196
398 1375 398 398 405 499 499 495 488 471
294 1256 296 296 296 296 287 286 286 247
 
*/

#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

const int N = 10;

/*

001~abc-1, 999

abc
    1. num[i] < x, 0
    2. num[i] == x, 0~efg
    3. num[i] > x, 0~999

*/

int get(vector<int> num, int l, int r)
{
    int res = 0;
    for (int i = l; i >= r; i -- ) res = res * 10 + num[i];
    return res;
}

int power10(int x)
{
    int res = 1;
    while (x -- ) res *= 10;
    return res;
}

// 1~n之间所有数字中，x的出现次数
int count(int n, int x)
{
    if (!n) return 0;
    
    // 123 -> 321
    vector<int> num;
    while (n)
    {
        num.push_back(n % 10);
        n /= 10;
    }
    n = num.size();
    
    int res = 0;
    for (int i = n - 1 - !x; i >= 0; i -- )
    {
        if (i < n - 1)
        {
            res += get(num, n - 1, i + 1) * power10(i);
            if (!x) res -= power10(i);
        }

        if (num[i] == x) res += get(num, i - 1, 0) + 1;
        else if (num[i] > x) res += power10(i);
    }

    return res;
}

int main()
{
    int a, b;
    while (cin >> a >> b , a)
    {
        if (a > b) swap(a, b);

        for (int i = 0; i <= 9; i ++ )
            cout << count(b, i) - count(a - 1, i) << ' ';
        cout << endl;
    }

    return 0;
}
```



状态压缩DP：***蒙德里安的梦想*** 



```
/*
预处理：
st[j|k]：不能连续奇数个零
j & k ：不能冲突

j: 前一列伸到本列的数量
j：i列中被伸到的数量
k：i-1列伸出的数量
*/

/*
I 随意组（n，m）
(n, m)尺寸的棋盘分割成(1, 2)尺寸的格子，多少种分法
O 分法数量

I
1 2
1 3
1 4
2 2
2 3
2 4
2 11
4 11
0 0

O
1
0
1
2
3
5
144
51205

*/

#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 12, M = 1 << N;

int n, m;
long long f[N][M];
bool st[M];

int main()
{
    while (cin >> n >> m, n || m)
    {
        for (int i = 0; i < 1 << n; i ++ )
        {
            int cnt = 0;
            st[i] = true;
            for (int j = 0; j < n; j ++ )
                if (i >> j & 1)
                {
                    if (cnt & 1) st[i] = false;
                    cnt = 0;
                }
                else cnt ++ ;
            if (cnt & 1) st[i] = false;
        }

        memset(f, 0, sizeof f);

        f[0][0] = 1;
        for (int i = 1; i <= m; i ++ )
            for (int j = 0; j < 1 << n; j ++ )
                for (int k = 0; k < 1 << n; k ++ )
                    if ((j & k) == 0 && st[j | k])
                        f[i][j] += f[i - 1][k];

        cout << f[m][0] << endl;
    }
    return 0;
}
```



状态压缩DP：***最短Hamilton路径*** 



```
/*
I n，n：aij
n节点的带权无向图，标号0～n-1，0到n-1的最短hamilton路径
hamilton路径是补充不漏的经过每个点恰好一次
O 路径长度

I
5
0 2 4 5 1
2 0 6 5 3
4 6 0 8 3
5 5 8 0 5
1 3 3 5 0

O
18

*/

/*
遍历状态：i=0~(1<<n - 1): j=0~(n - 1)
判断状态合理：目标节点走到j：i>>j & 1
遍历中间状态转移：k=0~(n - 1)
判断中间状态合理：i>>k & 1
*/

#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 20, M = 1 << N;

int n;
int w[N][N];
int f[M][N];

int main()
{
    cin >> n;
    for (int i = 0; i < n; i ++ )
        for (int j = 0; j < n; j ++ )
            cin >> w[i][j];

    memset(f, 0x3f, sizeof f);
    f[1][0] = 0;

    for (int i = 0; i < 1 << n; i ++ )
        for (int j = 0; j < n; j ++ )
            if (i >> j & 1)
                for (int k = 0; k < n; k ++ )
                    if (i >> k & 1)
                        f[i][j] = min(f[i][j], f[i - (1 << j)][k] + w[k][j]);

    cout << f[(1 << n) - 1][n - 1];

    return 0;
}
```



树形DP：***没有上司的舞会*** 



```
/*
I n，n：hi, n-1: li，ki
没有上司的舞会：n个职员，每个职员开心度hi，li的上司是ki
没有直接上司的情况下，可以邀请的职员开心度总和最大值
O 开心度总和最大值

I
7
1
1
1
1
1
1
1
1 3
2 3
6 4
7 4
4 5
3 5

O
5

*/

#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 6010;

int n;
int h[N], e[N], ne[N], idx;
int happy[N];
int f[N][2];
bool has_fa[N];

void add(int a, int b)
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

void dfs(int u)
{
    f[u][1] = happy[u];

    for (int i = h[u]; ~i; i = ne[i])
    {
        int j = e[i];
        dfs(j);

        f[u][1] += f[j][0];
        f[u][0] += max(f[j][0], f[j][1]);
    }
}

int main()
{
    scanf("%d", &n);

    for (int i = 1; i <= n; i ++ ) scanf("%d", &happy[i]);

    memset(h, -1, sizeof h);
    for (int i = 0; i < n - 1; i ++ )
    {
        int a, b;
        scanf("%d%d", &a, &b);
        add(b, a);
        has_fa[a] = true;
    }

    int root = 1;
    while (has_fa[root]) root ++ ;

    dfs(root);

    printf("%d\n", max(f[root][0], f[root][1]));

    return 0;
}
```











###### 六章 贪心



讲问题，讲证明，讲代码



贪心的代码看似好写：就是一个排序，但是证明正确性却很难
—— 比如区间点：为什么要那么排序、判断
—— 排序：为什么按照右端点，其实左右等价，按照右排就要看左
—— 判断：
—— 形象来看就是：右排看左，每次都收集完最左边的可用来串的点，然后看下一个区间再右边移动



贪心最简单方式：排序 + 判断



贪心问题虽然靠直觉，但是不能纯靠直觉，很多最优解有数学模型上的严格证明：
仅靠直觉，难以说服自己，也难以说服别人



区间排序：做排序和右排序其实是等价的，重点在于相对的处理形成的结构



区间问题：
	都跟区间排序有关：按照一个区间值进行排序，左端位置或者右端位置
	递增的顺序进行遍历，这就是贪心单轴的特点



y总使用堆：没有手动模拟
—— 思考：估计需要使用floyd建堆堆算法维护一个队从而手动实现



y总说贪心题：猜 + 证明
所谓猜：感觉一下，或者按照样例去感受一种可行的模式



***区间选点*** 

```
struct Range
{
    int l, r;
    bool operator< (const Range &W)const
    {
        return r < W.r;
    }
}range[N];

// 右端排序区间从而顺序遍历，左端判断是否包含点
    int res = 0, ed = -2 * 1e9;
    for (int i = 0; i < n; i ++ )
        if (range[i].l > ed)
        {
            res ++ ;
            ed = range[i].r;
        }
```



```
/*
I n， n：li，ri
数轴上选择最少量的点，使得每个区间都包含至少一个点
O 最少的点数

I
3
-1 1
2 4
3 5

O
2

*/

#include <iostream>
#include <algorithm>

using namespace std;

const int N = 100010;

int n;
struct Range
{
    int l, r;
    bool operator< (const Range &W)const
    {
        return r < W.r;
    }
}range[N];

int main()
{
    scanf("%d", &n);
    for (int i = 0; i < n; i ++ ) scanf("%d%d", &range[i].l, &range[i].r);

    sort(range, range + n);

    // 右端排序区间从而顺序遍历，左端判断是否包含点
    int res = 0, ed = -2 * 1e9;
    for (int i = 0; i < n; i ++ )
        if (range[i].l > ed)
        {
            res ++ ;
            ed = range[i].r;
        }

    printf("%d\n", res);

    return 0;
}
```



***最大不相交区间数量*** 

毫不相交的区间块

贪心问题很多时候：就是有一个单调的轴

贪心问题：最核心在于理解问题本身讲什么

```
/*
I n， n：li，ri
数轴上选择最少量的点，使得每个区间都包含至少一个点
O 最少的点数

I
3
-1 1
2 4
3 5

O
2

*/

#include <iostream>
#include <algorithm>

using namespace std;

const int N = 100010;

int n;
struct Range
{
    int l, r;
    bool operator< (const Range &W)const
    {
        return r < W.r;
    }
}range[N];

int main()
{
    scanf("%d", &n);
    for (int i = 0; i < n; i ++ ) scanf("%d%d", &range[i].l, &range[i].r);

    sort(range, range + n);

    // 右端排序区间从而顺序遍历，左端判断是否包含点
    int res = 0, ed = -2 * 1e9;
    for (int i = 0; i < n; i ++ )
        if (range[i].l > ed)
        {
            res ++ ;
            ed = range[i].r;
        }

    printf("%d\n", res);

    return 0;
}
```



***区间分组*** 

优先队列，分组思路判断相交

```
/*
I n， n：li， ri
区间分组，组内无交集，求最小分组数
O 最小组数

I 
3
-1 1
2 4
3 5

O
2

*/

#include <iostream>
#include <algorithm>
#include <queue>

using namespace std;

const int N = 100010;

struct Range
{
    int l, r;
    bool operator< (const Range &W)const
    {
        return l < W.l;
    }
}
range[N];

int main()
{
    int n;
    scanf("%d", &n);
    for (int i = 0; i < n; i ++ ) scanf("%d%d", & range[i].l, & range[i].r);
    
    sort(range, range + n);

    // 优先队列：存Max_r
    // 当max_r > l时，说明新的区间存在相交
    priority_queue<int, vector<int>, greater<int>> heap;
    for (int i = 0; i < n; i ++ )
    {
        auto r = range[i];
        if (heap.empty() || heap.top() >= r.l) heap.push(r.r);
        else
        {
            heap.pop();
            heap.push(r.r);
        }
    }

    printf("%d\n", heap.size());

    return 0;
}
```



***区间分组*** 



```
/*
I n， n：li， ri
区间分组，组内无交集，求最小分组数
O 最小组数

I 
3
-1 1
2 4
3 5

O
2

*/

#include <iostream>
#include <algorithm>
#include <queue>

using namespace std;

const int N = 100010;

// 排序方式变化：左端排序方式
struct Range
{
    int l, r;
    bool operator< (const Range &W)const
    {
        return l < W.l;
    }
}
range[N];

int main()
{
    int n;
    scanf("%d", &n);
    for (int i = 0; i < n; i ++ ) scanf("%d%d", & range[i].l, & range[i].r);

    sort(range, range + n);

    // 优先队列：存Max_r也就是一组中的w.r
    // 当max_r > l时，说明新的区间存在相交，否则不相交那么更新该组的最大右端点
    // 注意最大右端点是在递增遍历的
    priority_queue<int, vector<int>, greater<int>> heap;
    for (int i = 0; i < n; i ++ )
    {
        if (heap.empty() || heap.top() >= range[i].l) heap.push(range[i].r);
        else
        {
            heap.pop();
            heap.push(range[i].r);
        }
    }

    printf("%d\n", heap.size());

    return 0;
}
```



***区间覆盖*** 

果然：贪心问题的核心还是理解贪心问题本身
—— 理解得多了，还是能对每个题目进行具体化地理解的
—— 左端点排序，选能够覆盖目标区间的右端点最靠右的可选区间

```
/*
I s，t， n，n：li，ri
给定区间[s,t]，选择尽量少的区间，将[s,t]区间覆盖
O 最少所需区间数量

I
1 5
3
-1 3
2 4
3 5

O
2

*/


#include <iostream>
#include <algorithm>

using namespace std;

const int N = 100010;

int n;
struct Range
{
    int l, r;
    bool operator< (const Range &W)const
    {
        return l < W.l;
    }
}range[N];

int main()
{
    int st, ed;
    scanf("%d%d", &st, &ed);
    scanf("%d", &n);
    for (int i = 0; i < n; i ++ )
    {
        int l, r;
        scanf("%d%d", &l, &r);
        range[i] = {l, r};
    }

    sort(range, range + n);

    // 双指针算法(遍历的基础上，二次向后遍历)：每次选择一个区间
    // 判断：不符合条件和达到条件
    // 每轮更新：st，i = j-1表示当前的j区间（while中多自增了一下）
    int res = 0;
    bool success = false;
    for (int i = 0; i < n; i ++ )
    {
        int j = i, r = -2e9;
        while (j < n && range[j].l <= st)
        {
            r = max(r, range[j].r);
            j ++ ;
        }
        res ++ ;

        if (r < st)
        {
            res = -1;
            break;
        }
        if (r >= ed)
        {
            success = true;
            break;
        }

        st = r;
        i = j - 1;
    }

    if (!success) res = -1;
    printf("%d\n", res);

    return 0;
}
```



Hffuleman树：***合并果子*** 



```
/*
I n， n：ai
每堆果子的数量为ai，合并两堆果子消耗ai+aj的体力，合并所有的ai，使得消耗最小
O 最小体力消耗值

I
3
1 2 9

o
15
*/

#include <iostream>
#include <algorithm>
#include <queue>

using namespace std;

int main()
{
    int n;
    scanf("%d", &n);

    // 便捷地进行哈夫曼树选择：最小堆管理元素
    priority_queue<int, vector<int>, greater<int>> heap;
    while (n -- )
    {
        int x;
        scanf("%d", &x);
        heap.push(x);
    }

    int res = 0;
    while (heap.size() > 1)
    {
        int a = heap.top(); heap.pop();
        int b = heap.top(); heap.pop();
        res += a + b;
        heap.push(a + b);
    }

    printf("%d\n", res);
    return 0;
}
```



排序不等式：***排队打水*** 

```
/*
I n， n ： ti
n个人排队打水，每个人需要ti打水时间，安排打水顺序使得总体等待时间最小
O 最小等待时间之和

I
7
3 6 1 4 2 5 7

O
56

*/

#include <iostream>
#include <algorithm>

using namespace std;

// n < 10^5, res可到10^5^2 / 2 > 2 * 10^9(int范围)
using LL = long long;

const int N = 100010;

int n;
int t[N];

int main()
{
    scanf("%d", &n);
    for (int i = 0; i < n; i ++ ) scanf("%d", &t[i]);

    sort(t, t + n);
    reverse(t, t + n);

    LL res = 0;
    for (int i = 0; i < n; i ++ ) res += t[i] * i;

    printf("%lld\n", res);

    return 0;
}
```



绝对值不等式：***货仓选址*** 

```
/*
I n，n：ai
数轴上存在n个商店，位置为ai，需要建立一个仓库，求仓库位置，使得其到达所有商店的距离为最小
O 距离之和的最小值

I
4
6 2 9 1

O
12

*/


#include <iostream>
#include <algorithm>

using namespace std;

const int N = 100010;

int n;
int q[N];

int main()
{
    scanf("%d", &n);

    for (int i = 0; i < n; i ++ ) scanf("%d", &q[i]);

    sort(q, q + n);

    int res = 0;
    for (int i = 0; i < n; i ++ ) res += abs(q[i] - q[n / 2]);

    printf("%d\n", res);

    return 0;
}
```



推公式：***耍杂技的牛*** 

```
/*
I n，n：wi，si
n头牛叠罗汉，自身重wi，支撑力si，风险值=si-求和身上的wi，确定顺序，使得最大风险值为最小
O 最大风险值的数值

I
3
10 3
2 5
3 3

O
2

*/

#include <iostream>
#include <algorithm>

using namespace std;

typedef pair<int, int> PII;

const int N = 50010;

int n;
PII cow[N];

int main()
{
    scanf("%d", &n);
    for (int i = 0; i < n; i ++ )
    {
        int s, w;
        scanf("%d%d", &w, &s);
        cow[i] = {w + s, w};
    }

    sort(cow, cow + n);

    // 读懂题目即可：重量轻的尽量往上即可，s自然都会加上
    int res = -2e9, sum = 0;
    for (int i = 0; i < n; i ++ )
    {
        int s = cow[i].first - cow[i].second, w = cow[i].second;
        res = max(res, sum - s);
        sum += w;
    }

    printf("%d\n", res);

    return 0;
}
```



###### 七章 复杂度分析



～～ 三种复杂度：
时间：函数与过程相互嵌套
空间：全局和局部内存处理
数值：n=1~10^5，小心n^2/2就会爆int，50亿大于int的20亿