// Copyright 2013 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100 //一个词的最大长度
#define EXP_TABLE_SIZE 1000 //exp_table是对sigmoid函数的计算结果进行缓存，需要用时查表即可，节省计算时间，这里定义只能存储1000个结果
#define MAX_EXP 6 //计算sigmoid函数值时，e的指数最大是6，最小是-6
#define MAX_SENTENCE_LENGTH 1000 //最大的句子长度，指句中词的个数
#define MAX_CODE_LENGTH 40 //最长的huffman编码长度

//哈希表长，采用开放定址法解决冲突，负载因子为0.7，所以实际允许用于hash的词个数为vocab_hash_size * 0.7=2.1×10^7
const int vocab_hash_size = 30000000; // Maximum 30 * 0.7 = 21M words in the vocabulary
typedef float real; // Precision of float numbers

struct vocab_word 
{
	long long cn; //词频
	int *point; //huffman树中从根节点到当前词所对应的叶子节点的路径中，保存其中非叶子节点的索引
	char *word, *code, codelen; //当前词，其对应的huffman编码及编码长度
};

char train_file[MAX_STRING], output_file[MAX_STRING]; //训练文件名 和 输出文件名
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING]; //保存词表的文件 和 待从该文件中读入词表

struct vocab_word *vocab;//词表

/*其中的min_reduce是指在ReduceVocab函数中会删除词频小于这个值的词，因为哈希表总共可以装填的词汇数是有限的*/
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;

//词汇表的hash存储，下标是词的hash值，数组内容是词在词表中的位置
int *vocab_hash;//对词汇表进行哈希操作的指针

/*分别为词表的大小，词表的当前大小，input layer中每个word vector的维数(默认为100)*/
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;

/*分别为待训练的词总数(对于从已经做好词频统计工作的词表文件中读取并构造词表的过程中，该值就是词频累加，以此统计个数；
对于从单纯的训练数据中读取单词自己构造词表时，就需要每读一个词就进行++操作，统计词个数)
已训练的词个数，迭代次数，训练文件的大小(有ftell函数所得)，聚类时的类别个数*/
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;

/*其中starting_alpha用于保存学习速率的初始值，因为alpha会在模型训练的反向传播过程中自动调整*/
real alpha = 0.025, starting_alpha, sample = 1e-3;//学习速率初始化为0.025，这是对于skip-gram而言的

/*分别为input layer的词向量序列，softmax中hidden layer到huffman tree中非叶子节点的映射权重，negative sampling中hidden layer到分类问题的映射局权重，
保存sigmoid函数值的表*/
real *syn0, *syn1, *syn1neg, *expTable;

clock_t start;//程序开始时间

int hs = 0, negative = 5;//前者：是否采用softmax体系的标识；后者：负样本的数量
const int table_size = 1e8;//在负采样中用做等距离剖分
int *table;//采样表

//构建负采样算法中的权值分布表
void InitUnigramTable() 
{
	int a, i;//循环变量

	double train_words_pow = 0;//词的权重总值，用于做归一化处理
	double d1, power = 0.75;//对每个词定义权值的时候不是直接取count，而是计算了α次幂，此处α取做0.75

	table = (int *)malloc(table_size * sizeof(int));//为table分配空间

	for (a = 0; a < vocab_size; a++) //遍历词汇表，计算train_words_pow
		train_words_pow += pow(vocab[a].cn, power);

	i = 0;//遍历词表
	d1 = pow(vocab[i].cn, power) / train_words_pow;//初始化已遍历的词的权值占权重总值的比例，初始值为第一个词的权值在总体中的占比

	//遍历table，将等距剖分投影到以词表中词的权重值为基准进行的非等距剖分中，此时table的每个单元中保存的是其所投影到的非等距剖分的段序
	for (a = 0; a < table_size; a++)
	{
		table[a] = i;//当前table中的等距离剖分段a对应的是非等距剖分段i

		if (a / (double)table_size > d1) 
		{
			i++;//进入词表中的下一个位置
			d1 += pow(vocab[i].cn, power) / train_words_pow;//将当前词的权值占权重总值的比例加入到已累计词的当前总值中
		}

		if (i >= vocab_size) //判断是否超过词汇表的大小
			i = vocab_size - 1;
	}
}

//从文件中读取单个的词，假设空格+tab+EOL(end of line)是单词边界
//word用于保存读取到的单词
void ReadWord(char *word, FILE *fin) 
{
	int a = 0, ch;//a用于计数word中字符的位置，ch用于保存当前读取到的字符

	while (!feof(fin))//判断是否到达文件结尾 
	{
		ch = fgetc(fin);//读入单词

		if (ch == 13) //对应EOF，需要进入下一行
			continue;

		if ((ch == ' ') || (ch == '\t') || (ch == '\n'))//遇到单词边界 
		{
			if (a > 0) //word中已经保存了内容
			{
				if (ch == '\n') 
					ungetc(ch, fin);//将一个字符退回到输入流中，这个退回的字符会由下一个读取文件流的函数取得
				break;
			}

			if (ch == '\n') 
			{
				strcpy(word, (char *)"</s>");//添加sentence结束的标记
				return;
			} 
			else 
				continue;
		}

		word[a] = ch;
		a++;

		if (a >= MAX_STRING - 1)//截断过长的单词
			a--;
	}

	word[a] = 0;//加上结束符‘\0’
}

//返回一个词的hash值。一个词对应一个hash值，但一个hash值可以对应多个词，即哈希冲突
int GetWordHash(char *word) 
{
	unsigned long long a, hash = 0;
	
	for (a = 0; a < strlen(word); a++) //遍历word中的每个字符，采用257进制构造当前word的key value
		hash = hash * 257 + word[a];
	
	hash = hash % vocab_hash_size;//定义散列函数为H(key)=key % m，即除留余数法
	
	return hash;
}

//返回当前词在词表中的位置，如果不存在就返回-1
int SearchVocab(char *word) 
{
	unsigned int hash = GetWordHash(word);//获得当前词的hash值
	
	while (1) 
	{
		if (vocab_hash[hash] == -1)//当前hash值在hash表中不存在，即当前词不存在，返回-1
			return -1;
			
		//用当前利用hash值在词表中找到的词与当前词对比，如果相同，则表示当前被hash值对应正确
		if (!strcmp(word, vocab[vocab_hash[hash]].word)) 
			return vocab_hash[hash];
		
		hash = (hash + 1) % vocab_hash_size;//开放定址法
	}
	return -1;
}

// 从文件流中读取一个词，并返回这个词在词汇表中的位置
int ReadWordIndex(FILE *fin) 
{
	char word[MAX_STRING];
	
	ReadWord(word, fin);//从文件中读取单词
	
	if (feof(fin)) 
		return -1;
	
	return SearchVocab(word);
}

//将一个词添加到词表中  
int AddWordToVocab(char *word) 
{
	unsigned int hash, length = strlen(word) + 1;//当前词的hash值和length
	
	if (length > MAX_STRING) 
		length = MAX_STRING;
	
	vocab[vocab_size].word = (char *)calloc(length, sizeof(char));//为待加入的词的位置分配空间
	
	strcpy(vocab[vocab_size].word, word);
	vocab[vocab_size].cn = 0;//将其词频置为0
	vocab_size++;
	
	// Reallocate memory if needed
	/*如果当前词表的实际大小接近最大限制大小的话，考虑扩容，每次空大1000
	由于当前重新分配的内存空间比原来的大，所以原内容保留，新增加的空间不初始化*/
	if (vocab_size + 2 >= vocab_max_size) 
	{
		vocab_max_size += 1000;
		vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
	}
	
	hash = GetWordHash(word);//获取当前新加入的词的hash值
	while (vocab_hash[hash] != -1) //即当前位置已经存放了其他词，产生了冲突
		hash = (hash + 1) % vocab_hash_size;//使用开放地址法解决冲突
	
	//由词的hash值找到其所在词表的排序位置。即在哈希表中当前词的hash值所指向的单元中存放的是该词在词表中的位置下标
	vocab_hash[hash] = vocab_size - 1;
	
	return vocab_size - 1;//返回新加入的词在词表中的下标
}

//比较两个词词频的大小
int VocabCompare(const void *a, const void *b) 
{
	return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

//将词表按照词频进行排序
void SortVocab() 
{
	int a, size;
	unsigned int hash;
	
	// Sort the vocabulary and keep </s> at the first position
	/*qsort函数，&vocab[1]为待排序的数组的头指针(这里不是&vocab[0]是为了保持</s>一直在首位)，
	vocab_size - 1为待排序数组的大小，sizeof(struct vocab_word)为数组元素大小，VocabCompare为判断大小的函数的指针*/
	qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
	
	for (a = 0; a < vocab_hash_size; a++) //重新初始化hash表
		vocab_hash[a] = -1;
	
	size = vocab_size;//保留当前词表的原始大小
	train_words = 0;//待训练的词的词频加和值此时为0
	
	for (a = 0; a < size; a++) //遍历当前词表中有的所有词
	{
		//判断当前词的词频是否小于最小值，是的话就从词表中去除
		if ((vocab[a].cn < min_count) && (a != 0)) 
		{
			vocab_size--;
			free(vocab[a].word);
		} 
		else 
		{
			// 排序后需要重新计算hash查找
			hash=GetWordHash(vocab[a].word);
			while (vocab_hash[hash] != -1) //发生冲突
				hash = (hash + 1) % vocab_hash_size;//开放地址法
			vocab_hash[hash] = a;//hash中保存当前词在词表中的位置
			
			train_words += vocab[a].cn;
		}
	}
	
	vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));//追加空间
	
	//为二叉树结构分配空间
	for (a = 0; a < vocab_size; a++) 
	{
		vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
		vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
	}
}

//移除词频过小的词，缩减词表
void ReduceVocab() 
{
	int a, b = 0;
	unsigned int hash;

	for (a = 0; a < vocab_size; a++) //遍历词表
	{
		if (vocab[a].cn > min_reduce) 
		{
			vocab[b].cn = vocab[a].cn;
			vocab[b].word = vocab[a].word;
			b++;
		} 
		else 
			free(vocab[a].word);
	}
	
	vocab_size = b;//当前词表在缩减操作后的实际大小
	for (a = 0; a < vocab_hash_size; a++)
		vocab_hash[a] = -1;
	for (a = 0; a < vocab_size; a++)//重新构造缩减操作后的词表的hash表
	{
		hash = GetWordHash(vocab[a].word);
		while (vocab_hash[hash] != -1)
			hash = (hash + 1) % vocab_hash_size;
		vocab_hash[hash] = a;
	}
	
	fflush(stdout);
	min_reduce++;//注意会递增词频的阈值
}

//通过词频信息构建huffman树，词频高的词的huffman编码会更短
void CreateBinaryTree() 
{
	/*pos1，pos2用于保存在构建huffman树的过程中挑选的权重最小的节点在词表中的下标
	min1i,min2i用于保存构建过程中找到的最小权重*/
	long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
	char code[MAX_CODE_LENGTH];
	
	//保存在构建huffman树过程中用到的节点词频信息。因为若叶子节点有n个，则构建出的树中非叶子节点个数是n-1个，则共2*n+1个节点
	long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));//保存左右支的编码信息，二者中最小者的编码为0而次小者的编码为1
	long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));//保存父亲节点在词表中的位置信息
	
	for (a = 0; a < vocab_size; a++) //初始化当前叶子节点词频信息(即权重)
		count[a] = vocab[a].cn;
	for (a = vocab_size; a < vocab_size * 2; a++) //对于未知的非叶子节点的权重，全部以一个大数初始化
		count[a] = 1e15;
	
	pos1 = vocab_size - 1;
	pos2 = vocab_size;
	
	//构造huffman树
	for (a = 0; a < vocab_size - 1; a++)//遍历词表 
	{
		// 找出目前权值最小的两个节点 
		if (pos1 >= 0) //第一个权值最小的节点  
		{
			if (count[pos1] < count[pos2]) 
			{
				min1i = pos1;
				pos1--;
			} 
			else 
			{
				min1i = pos2;
				pos2++;
			}
		} 
		else 
		{
			min1i = pos2;
			pos2++;
		}
		
		if (pos1 >= 0)//第二个权值最小的节点  
		{
			if (count[pos1] < count[pos2]) 
			{
				min2i = pos1;
				pos1--;
			} 
			else 
			{
				min2i = pos2;
				pos2++;
			}
		} 
		else 
		{
			min2i = pos2;
			pos2++;
		}
		
		count[vocab_size + a] = count[min1i] + count[min2i];//将新构造的非叶子节点的权重加入到权重集合
		
		//为孩子节点保存当前父亲节点在词表中的位置
		parent_node[min1i] = vocab_size + a;
		parent_node[min2i] = vocab_size + a;
		
		binary[min2i] = 1;//节点编码为1，之前默认是0。  
	}
	
	//将构造好的huffman树赋给词表中的每个词
	for (a = 0; a < vocab_size; a++) 
	{
		b = a;//当期遍历到的词在词表中的位置下标
		i = 0;//记录回溯的分支个数
		
		while (1) 
		{
			code[i] = binary[b];
			point[i] = b;
			i++;
			
			//@todo 判断i是否大于code_len,大于就记录错误并退出
			
			b = parent_node[b];//继续向根节点回溯
			if (b == vocab_size * 2 - 2) //已经到达根节点
				break; 
		}
		
		vocab[a].codelen = i;
		vocab[a].point[0] = vocab_size - 2;//将根节点放入路径中
		for (b = 0; b < i; b++) 
		{
			vocab[a].code[i - b - 1] = code[b];
			vocab[a].point[i - b] = point[b] - vocab_size;
		}
	}
	
	//释放空间
	free(count);
	free(binary);
	free(parent_node);
}

//从训练文件中读取单词来构造词表
//此文件中没有统计每个单词的词频
//同时获得训练数据文件的大小
void LearnVocabFromTrainFile()
{
	char word[MAX_STRING];//从训练文件中读出的词
	FILE *fin;
	long long a, i;
	
	for (a = 0; a < vocab_hash_size; a++) //hash表初始化
		vocab_hash[a] = -1;
	
	fin = fopen(train_file, "rb");//打开文件
	if (fin == NULL) 
	{
		printf("ERROR: training data file not found!\n");
		exit(1);
	}
	
	vocab_size = 0;//词表当前实际大小为0
	AddWordToVocab((char *)"</s>");//保证</s>在最前面
	
	while (1) 
	{
		ReadWord(word, fin);//从训练文件中读取一个词
		if (feof(fin)) 
			break;
		train_words++;//待训练但词个数加1
		
		if ((debug_mode > 1) && (train_words % 100000 == 0)) 
		{
			printf("%lldK%c", train_words / 1000, 13);
			fflush(stdout);
		}
		
		i = SearchVocab(word);//返回该词在词表中的位置
		if (i == -1)//此时该词不存在于词表之中，就将其加入词表之中  
		{
			a = AddWordToVocab(word);
			vocab[a].cn = 1;
		} 
		else 
			vocab[i].cn++;//更新词频
		
		if (vocab_size > vocab_hash_size * 0.7) //如果词表太庞大，就缩减词表
			ReduceVocab();
	}
	
	SortVocab();//根据词频将词表排序
	
	if (debug_mode > 0) 
	{
		printf("Vocab size: %lld\n", vocab_size);
		printf("Words in train file: %lld\n", train_words);
	}
	
	file_size = ftell(fin);//获得训练文件的大小
	fclose(fin);
}

//保存词表
void SaveVocab() 
{
	long long i;
	FILE *fo = fopen(save_vocab_file, "wb");
	for (i = 0; i < vocab_size; i++) 
		fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
	fclose(fo);
}

//从文件中读取词表，该文件已经统计好了每个词的词频
//同时获得训练数据文件的大小
void ReadVocab() 
{
	long long a, i = 0;
	char c;
	char word[MAX_STRING];//用于保存当前从文件中读出的词
	
	FILE *fin = fopen(read_vocab_file, "rb");//打开词表文件
	if (fin == NULL) 
	{
		printf("Vocabulary file not found\n");
		exit(1);
	}
	
	//hash表初始化
	for (a = 0; a < vocab_hash_size; a++) 
		vocab_hash[a] = -1;
	
	vocab_size = 0;//当前词表的实际大小为0
	while (1) 
	{
		ReadWord(word, fin);//从文件中读出一个词
		if (feof(fin)) 
			break;
		
		a = AddWordToVocab(word);//将当前从文件中读取出的词加入到词表中，返回其在词表中的位置
		
		/*从stream流中连续读取能够匹配format格式的字符到参数列表中对应的变量里
		fin为文件指针，"%lld%c"是format表达式,要由“空格”、“非空格”及“转换符”组成，格式为%[*][width][modifiers]type
		表示从文件中读取lld大小的、能转换成%c的字符（也就是char或int），将其保存到后面的地址中
		接下来是与“format”中“转换符”对应变量地址的列表，两地址间用逗号隔开。*/
		fscanf(fin, "%lld%c", &vocab[a].cn, &c);
		
		i++;
	}
	
	SortVocab();//根据词频排序
	
	if (debug_mode > 0) 
	{
		printf("Vocab size: %lld\n", vocab_size);
		printf("Words in train file: %lld\n", train_words);
	}
	
	fin = fopen(train_file, "rb");//读取训练数据
	if (fin == NULL) 
	{
		printf("ERROR: training data file not found!\n");
		exit(1);
	}
	
	fseek(fin, 0, SEEK_END);
	file_size = ftell(fin);//获得当前训练数据文件的大小
	fclose(fin);
}

//网络模型初始化
//为他们分配空间，初始化各个参数。同时构建huffman树
void InitNet() 
{
	long long a, b;
	unsigned long long next_random = 1;
	
	//posix_memalign() 成功时会返回size字节的动态内存，并且这块内存的地址是alignment(这里是128)的倍数,此时所分配的内存大小为词表大小 * 其中每个词的维度 * 实数大小
	a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));//预对齐内存的分配
	if (syn0 == NULL) //动态内存分配失败
	{
		printf("Memory allocation failed\n"); 
		exit(1);
	}
	
	if (hs) //采用softmax体系
	{
		a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));//为huffman tree中非叶子节点的权重分配动态内存
		if (syn1 == NULL) 
		{
			printf("Memory allocation failed\n"); 
			exit(1);
		}
		
		for (a = 0; a < vocab_size; a++) 
		{
			for (b = 0; b < layer1_size; b++)
				syn1[a * layer1_size + b] = 0;//每个非叶子节点的权重信息在syn1数组中就是1行
		}
	}
	
	if (negative>0) //还有负样本的内容
	{
		a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
		if (syn1neg == NULL) 
		{
			printf("Memory allocation failed\n");
			exit(1);
		}
		
		for (a = 0; a < vocab_size; a++) 
		{
			for (b = 0; b < layer1_size; b++)
				syn1neg[a * layer1_size + b] = 0;
		}
	}
	
	for (a = 0; a < vocab_size; a++) //随机初始化input layer中的每个word vector ，其在syn0中就是1行
	{
		for (b = 0; b < layer1_size; b++) 
		{
		next_random = next_random * (unsigned long long)25214903917 + 11;
		syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
		}
	}
	
	CreateBinaryTree();//建立huffman树，对词表中的每个词进行编码
}

//模型训练线程，在执行线程之前要根据词频排序的词表对每个词求得huffman编码
void *TrainModelThread(void *id) 
{
	/*cw为当前中心词的上下文词个数
	word 向句中添加单词用，句子完成后表示句子中的当前单词(即中心词)在词表中的位置
	last_word 上一个单词在词表中的位置，辅助扫描窗口
	sentence_length 当前句子的长度
	sentence_position 当前中心词在当前句子中的位置*/
	long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
	
	/*word_count 已训练词总个数
	last_word_count 保存值，以便在训练语料长度超过某个值时输出信息
	sen 当前句子*/
	long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
	
	long long l1, l2, c, target, label, local_iter = iter;
	unsigned long long next_random = (long long)id;
	real f, g;
	clock_t now;
	
	real *neu1 = (real *)calloc(layer1_size, sizeof(real));//隐层结点
	real *neu1e = (real *)calloc(layer1_size, sizeof(real));//误差累计项
	FILE *fi = fopen(train_file, "rb");
	
	 //每个线程对应一段文本。根据线程id找到自己负责的文本的初始位置 
	fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
	while (1) 
	{
		if (word_count - last_word_count > 10000) //训练词个数超过某阈值(10000)
		{
			word_count_actual += word_count - last_word_count;//已训练词的个数
			last_word_count = word_count;
			
			if ((debug_mode > 1)) 
			{
				now=clock();
				printf("%cAlpha: %f Progress: %.2f%% Words/thread/sec: %.2fk ", 13, alpha,
					word_count_actual / (real)(iter * train_words + 1) * 100,
					word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
				fflush(stdout);
			}
			
			alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));//逐次降低学习速率
			
			if (alpha < starting_alpha * 0.0001) //当学习速率过小时需要进行纠正
				alpha = starting_alpha * 0.0001;
		}
		
		// 读入一个句子
		if (sentence_length == 0) 
		{
			while (1) 
			{
				word = ReadWordIndex(fi);//从文件流中读取一个词，并返回这个词在词表中的位置
				
				if (feof(fi)) 
					break;
				if (word == -1)
					continue;
				
				word_count++;
				if (word == 0) //读到句尾
					break;
				
				//下采样过程中会随机丢弃高频词的词，但要保持排序的一致
				if (sample > 0) 
				{
					real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
					next_random = next_random * (unsigned long long)25214903917 + 11;
					if (ran < (next_random & 0xFFFF) / (real)65536)
						continue;
				}
				
				sen[sentence_length] = word;
				sentence_length++;
				if (sentence_length >= MAX_SENTENCE_LENGTH) 
					break;
			}
			
			sentence_position = 0;//初始化为0表示取当前句中的第一个词为中心词
		}
		
		if (feof(fi) || (word_count > train_words / num_threads)) //当前线程应处理的部分结束
		{
			word_count_actual += word_count - last_word_count;
			local_iter--;//迭代次数递减
			if (local_iter == 0) //迭代iter次就结束
				break;
			
			word_count = 0;
			last_word_count = 0;
			sentence_length = 0;
			fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
			continue;//开始新的一轮迭代
		}
		
		word = sen[sentence_position];//中心词在词表中的位置 
		if (word == -1) 
			continue;
		
		for (c = 0; c < layer1_size; c++) //初始化隐层单元
			neu1[c] = 0;
		for (c = 0; c < layer1_size; c++) 
			neu1e[c] = 0;
		
		next_random = next_random * (unsigned long long)25214903917 + 11;
		b = next_random % window;
		//窗口大小等于window-b
		
		//训练CBOW模型
		if (cbow) 
		{ 
			//从input layer到 hidden layer 的映射
			cw = 0;
			for (a = b; a < window * 2 + 1 - b; a++) //扫描中心词的左右的几个词
			{
				if (a != window) //此时仍未扫描到当前中心词
				{
					c = sentence_position - window + a;//当前扫描到的上下文词在词表中的位置
					if (c < 0) 
						continue;
					if (c >= sentence_length) 
						continue;
					
					last_word = sen[c];//记录上一个词在词表中的位置，方便辅助窗口扫描
					if (last_word == -1) 
						continue;
					
					for (c = 0; c < layer1_size; c++) //layer1_size词向量的维度，默认值是100 
						neu1[c] += syn0[c + last_word * layer1_size];
					
					cw++;//上下文词数加1
				}
			}
			
			//此时隐藏层单元已经计算结束
			
			if (cw) 
			{
				for (c = 0; c < layer1_size; c++) //做平均
					neu1[c] /= cw;
				
				if (hs) 
				{
					for (d = 0; d < vocab[word].codelen; d++) //逐节点遍历当前中心词存储的huffman树中的路径
					{
						f = 0;
						l2 = vocab[word].point[d] * layer1_size;//当前中心词的遍历到的非叶子节点的权重的开始位置
						
						// Propagate hidden -> output
					    for (c = 0; c < layer1_size; c++) 
							f += neu1[c] * syn1[c + l2];//计算内积
						
						if (f <= -MAX_EXP) //内积不在范围内则直接丢弃 
							continue;
						else if (f >= MAX_EXP) 
							continue;
						else 
							f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];//内积之后sigmoid函数
						
						g = (1 - vocab[word].code[d] - f) * alpha;// 'g' is the gradient multiplied by the learning rate
						
						// Propagate errors output -> hidden，layer1_size是向量的维度
						for (c = 0; c < layer1_size; c++) //反向传播误差，从huffman树传到隐藏层
							neu1e[c] += g * syn1[c + l2];//把当前非叶子节点的误差传播给隐藏层
						
						// Learn weights hidden -> output,更新当前非叶子节点的向量
						for (c = 0; c < layer1_size; c++) 
							syn1[c + l2] += g * neu1[c];
					}
				}

				// NEGATIVE SAMPLING
				if (negative > 0) 
				{
					for (d = 0; d < negative + 1; d++) 
					{
						if (d == 0) 
						{
							target = word;//中心词
							label = 1;//正样本
						}
						else 
						{
						    next_random = next_random * (unsigned long long)25214903917 + 11;
						    target = table[(next_random >> 16) % table_size];
						
						    if (target == 0) 
							    target = next_random % (vocab_size - 1) + 1;
						    if (target == word) 
							    continue;
						
						    label = 0;//负样本
					    }
						
						l2 = target * layer1_size;
					    f = 0;
					    for (c = 0; c < layer1_size; c++) 
						    f += neu1[c] * syn1neg[c + l2];//内积，隐层不变
					
					    if (f > MAX_EXP)
						    g = (label - 1) * alpha;
					    else if (f < -MAX_EXP) 
						    g = (label - 0) * alpha;
					    else 
						    g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
					
					    for (c = 0; c < layer1_size; c++) 
						    neu1e[c] += g * syn1neg[c + l2];//隐层的误差
					    for (c = 0; c < layer1_size; c++) 
						    syn1neg[c + l2] += g * neu1[c];//更新负样本向量
				    }
				}
				
				// hidden -> in
				for (a = b; a < window * 2 + 1 - b; a++)//对input layer的word vector进行更新
				{
					if (a != window) //没有遇到当前中心词
					{
						c = sentence_position - window + a;//当前上下文词在词典中的位置
						if (c < 0) 
							continue;
						if (c >= sentence_length) 
							continue;
						
						last_word = sen[c];
						if (last_word == -1) 
							continue;
						
						for (c = 0; c < layer1_size; c++) 
							syn0[c + last_word * layer1_size] += neu1e[c];
					}
				}						
			}
		}
		else //训练skip-gram模型
		{ 
			for (a = b; a < window * 2 + 1 - b; a++)  //↓
			{
				if (a != window)//扫描上下文词 //↑
				{
					c = sentence_position - window + a;
					if (c < 0) 
						continue;
					if (c >= sentence_length)
						continue;
					
					last_word = sen[c];
					
					if (last_word == -1) 
						continue;
					
					l1 = last_word * layer1_size;//当前遍历到的上下文词的向量表示的开始位置
					for (c = 0; c < layer1_size; c++) //重新初始化
						neu1e[c] = 0;
					
					// HIERARCHICAL SOFTMAX
					if (hs) 
					{
						for (d = 0; d < vocab[word].codelen; d++) //逐节点遍历当前中心词存储的huffman树中的路径
						{
							f = 0;
							l2 = vocab[word].point[d] * layer1_size;//当前中心词的遍历到的非叶子节点的权重的开始位置
						
						    // Propagate hidden -> output
							for (c = 0; c < layer1_size; c++) 
								f += syn0[c + l1] * syn1[c + l2];//计算两个词向量的内积(没有隐层) 
							
							if (f <= -MAX_EXP) 
								continue;
							else if (f >= MAX_EXP) 
								continue;
							else 
								f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
							
							g = (1 - vocab[word].code[d] - f) * alpha;// 'g' is the gradient multiplied by the learning rate
							
							// Propagate errors output -> hidden
							for (c = 0; c < layer1_size; c++) 
								neu1e[c] += g * syn1[c + l2];//隐藏层的误差
							
							// Learn weights hidden -> output
							for (c = 0; c < layer1_size; c++) 
								syn1[c + l2] += g * syn0[c + l1];//更新非叶子节点权重表示
						}
					}
					
					// NEGATIVE SAMPLING
					if (negative > 0)
					{ 
				        for (d = 0; d < negative + 1; d++) 
						{
							if (d == 0) 
							{
								target = word;
								label = 1;
							}
							else 
							{
								next_random = next_random * (unsigned long long)25214903917 + 11;
								target = table[(next_random >> 16) % table_size];
								if (target == 0) 
									target = next_random % (vocab_size - 1) + 1;
								if (target == word) 
									continue;
								label = 0;
							}
							
							l2 = target * layer1_size;
							
							f = 0;
							for (c = 0; c < layer1_size; c++) 
								f += syn0[c + l1] * syn1neg[c + l2];
							if (f > MAX_EXP) 
								g = (label - 1) * alpha;
							else if (f < -MAX_EXP) 
								g = (label - 0) * alpha;
							else 
								g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
							
							for (c = 0; c < layer1_size; c++) 
								neu1e[c] += g * syn1neg[c + l2];
							for (c = 0; c < layer1_size; c++) 
								syn1neg[c + l2] += g * syn0[c + l1];
						}
					}
					
					// Learn weights input -> hidden
					for (c = 0; c < layer1_size; c++) 
						syn0[c + l1] += neu1e[c];//更新周围几个词语的向量
				}
			}
		}
		
		sentence_position++;//选择下一个中心词
		if (sentence_position >= sentence_length) 
		{
			sentence_length = 0;
			continue;
		}
	}
	
	fclose(fi);
	free(neu1);
	free(neu1e);
	pthread_exit(NULL);
}

//模型训练
void TrainModel() 
{
	long a, b, c, d;
	FILE *fo;
	
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	printf("Starting training using file %s\n", train_file);
	
	starting_alpha = alpha;//保存学习速率的初始值
	if (read_vocab_file[0] != 0) 
		ReadVocab(); //从文件读入词表
	else
		LearnVocabFromTrainFile();//从训练文件学习词汇  
	
	if (save_vocab_file[0] != 0) 
		SaveVocab();//保存词表
	if (output_file[0] == 0) 
		return;
	
	InitNet();//网络初始化
	if (negative > 0) 
		InitUnigramTable();
	
	start = clock();//获取训练的开始时间
	for (a = 0; a < num_threads; a++) //线程创建
		pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
		
	//以阻塞的方式等待指定的线程结束。当函数返回时，被等待线程的资源被收回。如果线程已经结束，那么该函数会立即返回。并且指定的线程必须是joinable的。	
	for (a = 0; a < num_threads; a++) 
		pthread_join(pt[a], NULL);
	
	//在这里，所有的线程都已经执行结束了
	
	fo = fopen(output_file, "wb");
	if (classes == 0) //不需要聚类，只需要输出词向量
	{
		// Save the word vectors
		fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
		
		for (a = 0; a < vocab_size; a++) //遍历词表
		{
			fprintf(fo, "%s ", vocab[a].word);
			if (binary)//二进制形式输出
			{
				for (b = 0; b < layer1_size; b++) 
					fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
			}				
			else
			{
				for (b = 0; b < layer1_size; b++) 
					fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
			}				
				
			fprintf(fo, "\n");
		}
	} 
	else //使用k-means进行聚类
	{
		// Run K-means on the word vectors
		int clcn = classes, iter = 10, closeid;
		int *centcn = (int *)malloc(classes * sizeof(int));//该类别的数量
		int *cl = (int *)calloc(vocab_size, sizeof(int));//词到类别的映射 
		real closev, x;
		real *cent = (real *)calloc(classes * layer1_size, sizeof(real));//质心数组 
		
		for (a = 0; a < vocab_size; a++) 
			cl[a] = a % clcn;
		for (a = 0; a < iter; a++) 
		{
			for (b = 0; b < clcn * layer1_size; b++) 
				cent[b] = 0;//质心清零
			for (b = 0; b < clcn; b++) 
				centcn[b] = 1;
			for (c = 0; c < vocab_size; c++) 
			{
				for (d = 0; d < layer1_size; d++) 
					cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
				centcn[cl[c]]++;//类别数量加1
			}
			
			for (b = 0; b < clcn; b++)//遍历所有类别 
			{
				closev = 0;
				for (c = 0; c < layer1_size; c++) 
				{
					cent[layer1_size * b + c] /= centcn[b];//均值，即求新的质心 
					closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
				}
				
				closev = sqrt(closev);
				for (c = 0; c < layer1_size; c++) 
					cent[layer1_size * b + c] /= closev;/*？？？对质心进行归一化？？？*/  
			}
			
			for (c = 0; c < vocab_size; c++)//对所有词语重新分类 
			{
				closev = -10;
				closeid = 0;
				for (d = 0; d < clcn; d++) 
				{
					x = 0;
					for (b = 0; b < layer1_size; b++) 
						x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];//内积
					if (x > closev) 
					{
						closev = x;
						closeid = d;
					}
				}
				cl[c] = closeid;
			}
		}
		
		// Save the K-means classes
		for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
		free(centcn);
		free(cent);
		free(cl);
	}
	fclose(fo);
}

//用于读取命令行参数,其中argc为argument count，argv为argument value
int ArgPos(char *str, int argc, char **argv) 
{
	int a;
	for (a = 1; a < argc; a++) //依次读取参数,跳过可执行文件名所在的位置
	{
		if (!strcmp(str, argv[a])) 
		{
			if (a == argc - 1) 
			{
				printf("Argument missing for %s\n", str);
			    exit(1);
			}
			
			return a;
		}
	}
	return -1;
}

//主函数
/*完成的内容包括：接收参数，分配空间，计算sigmoid函数值表*/
int main(int argc, char **argv) 
{
	int i;//遍历命令行参数
	
	if (argc == 1) //main函数此时没有提供其他参数，则输出提示
	{
		printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
		printf("Options:\n");
		printf("Parameters for training:\n");
		
		printf("\t-train <file>\n");//输入文件：已分词的语料 
		printf("\t\tUse text data from <file> to train the model\n");
		
		printf("\t-output <file>\n");//输出文件：词向量或者词聚类
		printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
		
		printf("\t-size <int>\n");//词向量的维度，默认值是100 
		printf("\t\tSet size of word vectors; default is 100\n");
		
		printf("\t-window <int>\n");//窗口大小，默认是5
		printf("\t\tSet max skip length between words; default is 5\n");
		
		printf("\t-sample <float>\n");//设定词出现频率的阈值，对于常出现的词会被随机下采样掉，默认值是10^3，有效范围为(0,10^5) 
		printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
		printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
		
		printf("\t-hs <int>\n");//是否采用softmax体系，默认是0，即不采用  
		printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
		
		printf("\t-negative <int>\n");//负样本的数量，默认是5，通常使用3-10。0表示不使用。
		printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
		
		printf("\t-threads <int>\n");//开启的线程数量，默认是12  
		printf("\t\tUse <int> threads (default 12)\n");
		
		printf("\t-iter <int>\n");//迭代次数
		printf("\t\tRun more training iterations (default 5)\n");
		
		printf("\t-min-count <int>\n");//最小阈值。对于出现次数少于该值的词，会被抛弃掉，默认值为5
		printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
		
		printf("\t-alpha <float>\n");//学习速率初始值，对于skip-gram默认是0.025，对于cbow是0.05 
		printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
		
		printf("\t-classes <int>\n");//输出词类别，而不是词向量。默认值为0，即输出词向量
		printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
		
		printf("\t-debug <int>\n");//debug模式，默认是2，表示在训练过程中会输出更多信息 
		printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
		
		printf("\t-binary <int>\n");//是否用binary模式保存数据，默认是0，表示否（一般不用这种形式）
		printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
		
		printf("\t-save-vocab <file>\n");//指定保存词表的文件
		printf("\t\tThe vocabulary will be saved to <file>\n");
		
		printf("\t-read-vocab <file>\n");//指定词表从该文件读取，而不是由训练数据重组
		printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
		
		printf("\t-cbow <int>\n");//是否采用CBOW算法。默认是1，表示采用CBOW算法。
		printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
		
		printf("\nExamples:\n");//工具使用样例 
		printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
		return 0;
	}
	
	output_file[0] = 0;
	save_vocab_file[0] = 0;
	read_vocab_file[0] = 0;
	
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);//atoi是字符串转整型的函数
	if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
	if (cbow) alpha = 0.05;//修改采用cbow模式时所采用的默认学习速率
	if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
	
	//为词表分配vocab_max_size个vocab_word类型大小的空间，即vocab_max_size*sizeof(vocab_word)，该空间的初始长度为0字节
	vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
	//因为hash表可认为是元素间可能存在空隙的线性表，所以分配空间大小为hash表长(vocab_hash_size) * int的大小
	vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
	expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));//为存储sigmoid函数计算结果的表分配空间
	
	/*计算sigmoid函数值由于MAX_EXP定义为6，所以最小值为指数为-6时，即exp^-6/(1+exp^-6)=1/(1+exp^6);
	最大值为指数为6时，即exp^6/(1+exp^6)=1/(1+exp^-6)*/
	for (i = 0; i < EXP_TABLE_SIZE; i++) 
	{
		//expTable[i] = exp((i/ 1000 * 2-1) * 6) 即 e^-6 ~ e^6 
		expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
		//expTable[i] = 1/(1+e^6) ~ 1/(1+e^-6)即 0.01 ~ 1 的样子
		expTable[i] = expTable[i] / (expTable[i] + 1); // Precompute f(x) = x / (x + 1)
	}
	
	TrainModel();
	return 0;
}