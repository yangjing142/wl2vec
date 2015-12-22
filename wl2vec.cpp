#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100 //词的最大长度
#define EXP_TABLE_SIZE 1000 //expTable能存储的结果个数
#define MAX_EXP 6 //计算sigmoid函数值时，e的指数最大是6，最小是-6
#define MAX_SENTENCE_LENGTH 1000 //最大的句子长度，指句中词的个数
#define MAX_CODE_LENGTH 40 //最长的huffman编码长度

const int vocab_hash_size = 30000000;//哈希表长, Maximum 30 * 0.7 = 21M words in the vocabulary
typedef float real;//实数类型

//词表中单个词的数据结构
struct vocab_word 
{
	long long cn; //词频
	long long procn;//带权词频
	int *point; //huffman树中从根节点到当前词所对应的叶子节点的路径中，保存其中非叶子节点的索引
	char *word, *code, codelen; //当前词，其对应的huffman编码及编码长度
};

char trainFile[MAX_STRING], outputFile[MAX_STRING]; 
char vocabSaveFile[MAX_STRING], vocabReadFile[MAX_STRING]; 

struct vocab_word *vocabulary;//词表

//其中的min_reduce是指在缩小词表操作中会删除词频小于这个值的词
int cbow = 1, debug_mode = 2, min_count = 5, num_threads = 12, min_reduce = 1;

int *hashTable;//词汇表的hash存储

//分别为词表的大小，词表的当前实际大小，input layer中每个word vector的维数
long long vocab_max_size = 1000, vocab_size = 0, dimension = 100;

//分别为待训练的词总数,已训练的词个数，迭代次数，训练文件的大小
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0;

real alpha = 0.025, starting_alpha, sample = 1e-3;

/*分别为input layer的词向量序列，softmax中hidden layer到huffman tree中非叶子节点的映射权重，negative sampling中hidden layer到output layer的映射权重，保存sigmoid函数值的表*/
real *vector, *mapVec, *mapVecNeg, *expTable;

clock_t start;//程序开始时间

int hs = 0, negative = 5;//前者：是否采用softmax体系的标识；后者：负样本的数量
const int table_size = 1e8;//采样表的大小，在negative sampling中会用到
int *table;//采样表

//返回一个词的hash值
int getWordHash(char *word) 
{
	unsigned long long i;//迭代变量
	unsigned long long hash = 0;
	
	for (i = 0; i < strlen(word); i++) //遍历word中的每个字符，采用257进制构造当前word的key value
		hash = hash * 257 + word[i];
	
	hash = hash % vocab_hash_size;//除留余数法
	
	return hash;
}


//移除词频过小的词，缩减词表
void reduceVocab() 
{
	int i, j = 0;
	unsigned int hash;

	for (i = 0; i < vocab_size; i++) //遍历词表
	{
		if (vocabulary[i].cn > min_reduce) 
		{
			vocabulary[j].cn = vocabulary[i].cn;
			vocabulary[j].word = vocabulary[i].word;
			j++;
		} 
		else 
			free(vocabulary[i].word);
	}
	
	vocab_size = j;//当前词表在缩减操作后的实际大小
	for (i = 0; i < vocab_hash_size; i++)
		hashTable[i] = -1;
	for (i = 0; i < vocab_size; i++)//重新构造缩减操作后的词表的hash表
	{
		hash = getWordHash(vocabulary[i].word);
		while (hashTable[hash] != -1)
			hash = (hash + 1) % vocab_hash_size;
		hashTable[hash] = i;
	}
	
	fflush(stdout);
	min_reduce++;//注意会递增词频的阈值
}

//返回当前词在词表中的位置，如果不存在就返回-1
int searchVocab(char *word) 
{
	unsigned int hash = getWordHash(word);//获得当前词的hash值
	
	while (1) 
	{
		if (hashTable[hash] == -1)//当前hash值在hash表中不存在，即当前词不存在，返回-1
			return -1;
			
		//用当前利用hash值在词表中找到的词与当前词对比，如果相同，则表示当前被hash值对应正确
		if (!strcmp(word, vocabulary[hashTable[hash]].word)) 
			return hashTable[hash];
		
		hash = (hash + 1) % vocab_hash_size;//开放定址法
	}
	return -1;
}

//比较两个词词频的大小
int vocabCompare(const void *a, const void *b) 
{
	return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

//将词表按照词频进行排序
void sortVocab() 
{
	int i, size;
	unsigned int hash;
	
	// Sort the vocabulary and keep </s> at the first position
	qsort(&vocabulary[1], vocab_size - 1, sizeof(struct vocab_word), vocabCompare);
	
	for (i = 0; i < vocab_hash_size; i++) //重新初始化hash表
		hashTable[i] = -1;
	
	size = vocab_size;//保留当前词表的原始大小
	train_words = 0;//待训练的词的词频加和值此时为0
	
	for (i = 0; i < size; i++) //遍历当前词表中有的所有词
	{
		//判断当前词的词频是否小于最小值，是的话就从词表中去除
		if ((vocabulary[i].cn < min_count) && (i != 0)) 
		{
			vocab_size--;
			free(vocabulary[i].word);
		} 
		else 
		{
			// 排序后需要重新计算hash查找
			hash=getWordHash(vocabulary[i].word);
			while (hashTable[hash] != -1)
				hash = (hash + 1) % vocab_hash_size;//开放地址法解决冲突
			hashTable[hash] = i;
			
			train_words += vocabulary[i].cn;
		}
	}
	
	vocabulary = (struct vocab_word *)realloc(vocabulary, (vocab_size + 1) * sizeof(struct vocab_word));//追加空间
	
	//为二叉树结构分配空间
	for (i = 0; i < vocab_size; i++) 
	{
		vocabulary[i].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
		vocabulary[i].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
	}
}

//将一个词添加到词表中  
int addWordToVocab(char *word) 
{
	unsigned int hash, length = strlen(word) + 1;//当前词的hash值和length
	
	if (length > MAX_STRING) 
		length = MAX_STRING;
	
	vocabulary[vocab_size].word = (char *)calloc(length, sizeof(char));//为待加入的词的位置分配空间
	
	strcpy(vocabulary[vocab_size].word, word);
	vocabulary[vocab_size].cn = 0;//将其词频置为0
	vocab_size++;
	
	// Reallocate memory if needed
	if (vocab_size + 2 >= vocab_max_size) 
	{
		vocab_max_size += 1000;
		vocabulary = (struct vocab_word *)realloc(vocabulary, vocab_max_size * sizeof(struct vocab_word));
	}
	
	hash = getWordHash(word);//获取当前新加入的词的hash值
	while (hashTable[hash] != -1)
		hash = (hash + 1) % vocab_hash_size;//使用开放地址法解决冲突
	
	//在哈希表中保存当前词在词表中的位置
	hashTable[hash] = vocab_size - 1;
	
	return vocab_size - 1;//返回新加入的词在词表中的下标
}

//从文件中读取单个的词
void readWord(char *word, FILE *fin) 
{
	int index = 0;//计数当前读取到的字符在word中的位置
	int ch;//保存当前读取到的字符

	while (!feof(fin))//判断是否到达文件结尾 
	{
		ch = fgetc(fin);//读入单词

		if (ch == '\n') //需要进入下一行
			continue;

		if ((ch == ' ') || (ch == '\t') || (ch == '\n'))//遇到单词边界 
		{
			if (index > 0) //word中已经保存了内容
			{
				if (ch == '\n') 
					ungetc(ch, fin);//将一个字符退回到输入流中，这个退回的字符会由下一个读取文件流的函数取得
				break;
			}

			if (ch == '\n') 
				continue;
		}

		word[index] = ch;
		index++;

		if (index >= MAX_STRING - 1)//截断过长的单词
			index--;
	}

	word[index] = 0;//加上结束符‘\0’
}

//从训练文件中读取一个词(用于构建词表)，空格为单词边界
void readWordFromFile(char *word,FILE *fin) 
{
	int index=0;//计数word中字符的位置
	char ch;//保存当前读取到的字符

	while (!feof(fin))//判断是否到达文件结尾 
	{
		ch = fgetc(fin);//读入当前字符
		while(ch!=' ')//在遇到作为单词边界的空格之前都是当前词
		{
			if(ch=='*')
			{
				ch = fgetc(fin);
				break;
			}

			word[index]=ch;
			index++;

			if (index >= MAX_STRING - 1)//截断过长的单词
				index--;

			ch = fgetc(fin);
		}

		word[index] = 0;//加上结束符‘\0’
		/*if(index==0)//当前word中没有保存有效词
		{
			ch = fgetc(fin);
			break;
		}*/

		//当前遇到了单词边界，继续向后读知道遇到回车表示换行
		while(ch!='\n')
			ch = fgetc(fin);

		//此时文件指针到达下一行
		break;
	}
}

//读取词表
void readVocab() 
{
	long long i = 0;
	char c;//用于读取词频
	char word[MAX_STRING];//用于保存当前从文件中读出的词
	
	FILE *fin = fopen(vocabReadFile, "rb");//打开词表文件
	if (fin == NULL) 
	{
		printf("Vocabulary file not found\n");
		exit(1);
	}
	
	//hash表初始化
	for (i = 0; i < vocab_hash_size; i++) 
		hashTable[i] = -1;
	
	vocab_size = 0;//当前词表的实际大小为0
	while (1) 
	{
		readWord(word, fin);//从文件中读出一个词
		if (feof(fin)) 
			break;
		
		i = addWordToVocab(word);//将当前从文件中读取出的词加入到词表中，返回其在词表中的位置
		fscanf(fin, "%lld%c", &vocabulary[i].cn, &c);//读取词频
		
		vocab_size++;
	}
	
	sortVocab();//根据词频排序
	
	if (debug_mode > 0) 
	{
		printf("Vocab size: %lld\n", vocab_size);
		printf("Words in train file: %lld\n", train_words);
	}
	
	fin = fopen(trainFile, "rb");
	if (fin == NULL) 
	{
		printf("ERROR: training data file not found!\n");
		exit(1);
	}
	
	fseek(fin, 0, SEEK_END);
	file_size = ftell(fin);//获得当前训练数据文件的大小
	fclose(fin);
}

//从训练文件中读取单词来构造词表
void learnVocabFromTrainFile()
{
	char word[MAX_STRING];//从训练文件中读出的词
	FILE *fin;
	long long i, index;
	
	for (i = 0; i < vocab_hash_size; i++) //hash表初始化
		hashTable[i] = -1;
	
	fin = fopen(trainFile, "rb");//打开文件
	if (fin == NULL) 
	{
		printf("ERROR: training data file not found!\n");
		exit(1);
	}
	
	vocab_size = 0;//词表当前实际大小为0
	addWordToVocab((char *)"</s>");//保证</s>在最前面
	
	while (1) 
	{
		readWordFromFile(word, fin);//从训练文件中读取一个词
		if (feof(fin)) 
			break;
		train_words++;//待训练但词个数加1
		
		if ((debug_mode > 1) && (train_words % 100000 == 0)) 
		{
			printf("%lldK%c", train_words / 1000, 13);
			fflush(stdout);
		}
		
		index = searchVocab(word);//返回该词在词表中的位置
		if (index == -1)//此时该词不存在于词表之中，就将其加入词表之中  
		{
			i = addWordToVocab(word);
			vocabulary[i].cn = 1;
		} 
		else 
			vocabulary[index].cn++;//更新词频
		
		if (vocab_size > vocab_hash_size * 0.7) //如果词表太庞大，就缩减词表
			reduceVocab();
	}
	
	sortVocab();//根据词频将词表排序
	
	if (debug_mode > 0) 
	{
		printf("Vocab size: %lld\n", vocab_size);
		printf("Words in train file: %lld\n", train_words);
	}
	
	file_size = ftell(fin);//获得训练文件的大小
	fclose(fin);
}

//保存词表
void saveVocab() 
{
	long long i;
	FILE *fo = fopen(vocabSaveFile, "wb");
	for (i = 0; i < vocab_size; i++) 
		fprintf(fo, "%s %lld\n", vocabulary[i].word, vocabulary[i].cn);
	fclose(fo);
}

//通过词频信息构建huffman树，词频高的词的huffman编码会更短
void createBinaryTree() 
{
	long long i,j;
	long long index;
	long long position_1,position_2;//在构建huffman树的过程中挑选的权重最小的节点在词表中的下标
	long long min_1,min_2;//构建过程中找到的最小权重

	long long point[MAX_CODE_LENGTH];//从根节点到当前词的路径
	char code[MAX_CODE_LENGTH];//当前词的huffman code
	
	//保存在构建huffman树过程中用到的节点词频信息
	long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));

	//保存左右支的编码信息，二者中最小者的编码为0而次小者的编码为1
	long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));

	//保存父亲节点在词表中的位置信息
	long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	
	for (i = 0; i < vocab_size; i++) //初始化当前叶子节点词频信息(即权重)
		count[i] = vocabulary[i].cn;
	for (i = vocab_size; i < vocab_size * 2; i++) //对于未知的非叶子节点的权重，全部以一个大数初始化
		count[i] = 1e15;
	
	position_1 = vocab_size - 1;
	position_2 = vocab_size;
	
	//构造huffman树
	for (i = 0; i < vocab_size - 1; i++)//遍历词表 
	{
		// 找出目前权值最小的两个节点 
		if (position_1 >= 0) //第一个权值最小的节点  
		{
			if (count[position_1] < count[position_2]) 
			{
				min_1 = position_1;
				position_1--;
			} 
			else 
			{
				min_1 = position_2;
				position_2++;
			}
		} 
		else 
		{
			min_1 = position_2;
			position_2++;
		}
		
		if (position_1 >= 0)//第二个权值最小的节点  
		{
			if (count[position_1] < count[position_2]) 
			{
				min_2 = position_1;
				position_1--;
			} 
			else 
			{
				min_2 = position_2;
				position_2++;
			}
		} 
		else 
		{
			min_2 = position_2;
			position_2++;
		}
		
		count[vocab_size + i] = count[min_1] + count[min_2];//将新构造的非叶子节点的权重加入到权重集合
		
		//为孩子节点保存当前父亲节点在词表中的位置
		parent_node[min_1] = vocab_size + i;
		parent_node[min_2] = vocab_size + i;
		
		binary[min_2] = 1;//节点编码为1，之前默认是0 
	}
	
	//将构造好的huffman树赋给词表中的每个词
	for (i = 0; i < vocab_size; i++) 
	{
		j = i;//当期遍历到的词在词表中的位置下标
		index = 0;//记录回溯的分支个数
		
		while (1) 
		{
			code[index] = binary[j];
			point[index] = j;
			index++;

			if(index>MAX_CODE_LENGTH)//当前遍历的深度超过规定的最大深度就退出
			{
				printf("Current index has exceeded the max_code_length!\n");
				exit(1);
			}
			
			j = parent_node[j];//继续向根节点回溯
			if (j == vocab_size * 2 - 2) //已经到达根节点
				break; 
		}
		
		vocabulary[i].codelen = index;
		vocabulary[i].point[0] = vocab_size - 2;//将根节点放入路径中
		for (j = 0; j < index; j++) 
		{
			vocabulary[i].code[index - j - 1] = code[j];
			vocabulary[i].point[index - j] = point[j] - vocab_size;
		}
	}
	
	//释放空间
	free(count);
	free(binary);
	free(parent_node);
}

//网络模型初始化,包括构建huffman树
void initNet() 
{
	long long i, j;
	unsigned long long next_random = 1;
	
	posix_memalign((void **)&vector, 128, (long long)vocab_size * dimension * sizeof(real));//预对齐内存的分配
	if (vector == NULL) //动态内存分配失败
	{
		printf("Memory allocation failed\n"); 
		exit(1);
	}
	
	if (hs) //采用softmax体系
	{
		//为huffman tree中非叶子节点的权重映射分配动态内存
		posix_memalign((void **)&mapVec, 128, (long long)vocab_size * dimension * sizeof(real));
		if (mapVec == NULL) 
		{
			printf("Memory allocation failed\n"); 
			exit(1);
		}
		
		for (i = 0; i < vocab_size; i++) 
		{
			for (j = 0; j < dimension; j++)
				mapVec[i * dimension + j] = 0;//每个非叶子节点的权重在mapVec中是dimension个单元,顺序存放
		}
	}
	
	if (negative>0) //negative sampleing体系
	{
		posix_memalign((void **)&mapVecNeg, 128, (long long)vocab_size * dimension * sizeof(real));//预对齐内存分配
		if (mapVecNeg == NULL) 
		{
			printf("Memory allocation failed\n");
			exit(1);
		}
		
		for (i = 0; i < vocab_size; i++) 
		{
			for (j = 0; j < dimension; j++)
				mapVecNeg[i * dimension + j] = 0;//每个负样本的表示在mapVecNeg中是dimension个单元,顺序存放
		}
	}
	
	for (i = 0; i < vocab_size; i++) //随机初始化input layer中的每个word vector
	{
		for (j = 0; j < dimension; j++) 
		{
		next_random = next_random * (unsigned long long)25214903917 + 11;
		vector[i * dimension + j] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / dimension;
		}
	}
	
	createBinaryTree();//建立huffman树，对词表中的每个词进行编码
}

//构建负采样算法中的权值分布表
void initUnigramTable() 
{
	int i, j;//循环变量

	double train_words_pow = 0;//词的权重总值，用于做归一化处理
	double len, power = 0.75;//对每个词定义权值的时候不是直接取count，而是计算了α次幂，此处α取做0.75

	table = (int *)malloc(table_size * sizeof(int));//为table分配空间

	for (i = 0; i < vocab_size; i++) //遍历词汇表，计算train_words_pow
		train_words_pow += pow(vocabulary[i].cn, power);

	j = 0;//遍历词表
	len = pow(vocabulary[j].cn, power) / train_words_pow;//初始化已遍历的词的权值占权重总值的比例，初始值为第一个词的权值在总体中的占比

	//遍历table，将等距剖分投影到以词表中词的权重值为基准进行的非等距剖分中
	for (i = 0; i < table_size; i++)
	{
		table[i] = j;//当前table中的等距离剖分段i对应的是非等距剖分段j

		if (i / (double)table_size > len) 
		{
			j++;//进入词表中的下一个位置
			len += pow(vocabulary[j].cn, power) / train_words_pow;//将当前词的权值占权重总值的比例加入到已累计词的当前总值中
		}

		if (j >= vocab_size) //判断是否超过词汇表的大小
			j = vocab_size - 1;
	}
}

//在训练模型时为读入一个句子而从文件中读入词或其对应的概率
//在线程开始处以保证读入将是完整的行
void readWordForSen(char *content, FILE *fin)
{
	int index = 0;//计数word中字符的位置
	char ch;//保存当前读取到的字符

	while (!feof(fin))//判断是否到达文件结尾 
	{
		ch = fgetc(fin);//读入当前字符
		
		if(feof(fin))
			break;
		/*
		if (ch == '*') //对应文件尾
		{
			ch = fgetc(fin);
			break;
		}*/

		if(ch=='\n')//遇到'\n',则将进入新的一行再后续处理
			continue;

		/*
		if(begin)
		{
			while(ch!='\n')//为防止不完整的句子被使用，一律在遇到第一个换行符之后再进行新句子的读入
				ch = fgetc(fin);//读入当前字符
		}*/

		if ((ch == ' ') || (ch == '\t') || (ch == '\n'))//遇到边界 
		{
			if (index > 0) //word中已经保存了内容
			{
				if (ch == '\n') 
					ungetc(ch, fin);//将一个字符退回到输入流中，这个退回的字符会由下一个读取文件流的函数取得
				break;
			}

			if (ch == '\n') 
			{
				strcpy(content, (char *)"</s>");//添加sentence结束的标记
				return;
			} 
			else 
				continue;
		}

		content[index] = ch;
		index++;

		if (index >= MAX_STRING - 1)//截断过长的单词
			index--;
	}

	content[index] = 0;//加上结束符‘\0’
}

//传统CBOW模型的训练
void trainTraditionalCBOW(long long *sen, long long length, real *hiddenVec, real *hiddenVecE, unsigned long long next_random,long long position)
{
	long long i,j;

	int cw=0;//当前中心词的上下文词个数
	
	long long cwPos=0;//当前上下文词在词表中的位置下标
	long long startPos=0;//当前遍历到的非叶子节点的权重在mapVec中的开始位置

	long long target, label;

	real innerProd=0;//内积
	real gradient=0;//梯度
	
	//从input layer到 hidden layer 的映射
	for ( i= 1; i < length; i++) //扫描当前中心词的上下文词
	{
		cwPos = sen[i];//当前上下文词在词表中的位置下标

		for (j = 0; j < dimension; j++) //计算hidden layer的内容，即input layer的
			hiddenVec[j] += vector[j + cwPos * dimension];

		cw++;
	}

	for (i = 0; i < dimension; i++) //对隐层单元做平均处理
		hiddenVec[i] /= cw;

	if(cw)
	{
		if (hs)//采用层次softmax体系 
		{
			for (i = 0; i < vocabulary[position].codelen; i++) //逐节点遍历当前中心词存储的huffman树中的路径
			{
				innerProd = 0;//内积
				startPos = vocabulary[position].point[i] * dimension;//当前遍历到的非叶子节点的权重在mapVec中的开始位置

				// Propagate hidden -> output
				for (j = 0; j < dimension; j++) 
					innerProd += hiddenVec[j] * mapVec[j + startPos];//计算内积

				if (innerProd <= -MAX_EXP) //内积不在范围内则直接丢弃 
					continue;
				else if (innerProd >= MAX_EXP) 
					continue;
				else 
					innerProd = expTable[(int)((innerProd + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];//内积之后sigmoid函数

				gradient = (1 - vocabulary[position].code[i] - innerProd) * alpha;// 这里的梯度，实际上是梯度和学习速率的乘积

				// Propagate errors output -> hidden
				for (j = 0; j < dimension; j++) //反向传播误差，从huffman树传到隐藏层
					hiddenVecE[j] += gradient * mapVec[j + startPos];//把当前非叶子节点的误差传播给隐藏层

				// Learn weights hidden -> output,更新当前非叶子节点的向量
				for (j = 0; j < dimension; j++) 
					mapVec[j + startPos] += gradient * hiddenVec[j];
			}
		}

		if(negative > 0)//采用neg体系
		{
			for (i = 0; i < negative + 1; i++) 
			{
				if (i == 0) 
				{
					target = position;//中心词
					label = 1;//正样本
				}
				else 
				{
					next_random = next_random * (unsigned long long)25214903917 + 11;
					target = table[(next_random >> 16) % table_size];

					if (target == 0) 
						target = next_random % (vocab_size - 1) + 1;
					if (target == position) 
						continue;

					label = 0;//负样本
				}

				startPos = target * dimension;//负样本信息在mapVecNeg中的开始位置

				for (j = 0; j < dimension; j++) 
					innerProd += hiddenVec[j] * mapVecNeg[j + startPos];//计算内积，隐层不变

				if (innerProd > MAX_EXP)
					gradient = (label - 1) * alpha;
				else if (innerProd < -MAX_EXP) 
					gradient = (label - 0) * alpha;
				else 
					gradient = (label - expTable[(int)((innerProd + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;//这里的梯度，实际上是梯度和学习速率的乘积

				for (j = 0; j < dimension; j++) 
					hiddenVecE[j] += gradient * mapVecNeg[j + startPos];//隐层的误差
				for (j = 0; j < dimension; j++) 
					mapVecNeg[j + startPos] += gradient * hiddenVec[j];//更新负样本向量
			}
		}

		// hidden -> in
		for (i = 1; i < length; i++)//对input layer的word vector进行更新
		{
			cwPos = sen[i];//当前上下文词在词表中的位置下标

			for (j = 0; j < dimension; j++) 
				vector[j + cwPos * dimension] += hiddenVecE[j];
		}
	}
}

//基于词图的CBOW模型的训练
void trainWordLatticeCBOW(long long *sen, real *senPro, long long length, real *hiddenVec, real *hiddenVecE, unsigned long long next_random,long long position)
{
	long long i,j;

	int cw=0;//当前中心词的上下文词个数
	
	long long cwPos=0;//当前上下文词在词表中的位置下标
	long long startPos=0;//当前遍历到的非叶子节点的权重在mapVec中的开始位置

	long long target, label;

	real innerProd=0;//内积
	real gradient=0;//梯度

	//从input layer到 hidden layer 的映射
	for ( i= 1; i < length; i++) //扫描当前中心词的上下文词
	{
		cwPos = sen[i];//当前上下文词在词表中的位置下标

		for (j = 0; j < dimension; j++) //计算hidden layer的内容，即input layer的
			hiddenVec[j] += vector[j + cwPos * dimension] * senPro[cwPos];

		cw++;
	}

	for (i = 0; i < dimension; i++) //对隐层单元做平均处理
		hiddenVec[i] /= cw;

	if(cw)
	{
		if (hs)//采用层次softmax体系 
		{
			for (i = 0; i < vocabulary[position].codelen; i++) //逐节点遍历当前中心词存储的huffman树中的路径
			{
				startPos = vocabulary[position].point[i] * dimension;//当前遍历到的非叶子节点的权重在mapVec中的开始位置

				// Propagate hidden -> output
				for (j = 0; j < dimension; j++) 
					innerProd += hiddenVec[j] * mapVec[j + startPos];//计算内积

				if (innerProd <= -MAX_EXP) //内积不在范围内则直接丢弃 
					continue;
				else if (innerProd >= MAX_EXP) 
					continue;
				else 
					innerProd = expTable[(int)((innerProd + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];//内积之后sigmoid函数

				gradient = (1 - vocabulary[position].code[i] - innerProd) * alpha;// 这里的梯度，实际上是梯度和学习速率的乘积

				// Propagate errors output -> hidden
				for (j = 0; j < dimension; j++) //反向传播误差，从huffman树传到隐藏层
					hiddenVecE[j] += gradient * mapVec[j + startPos] * senPro[0];//把当前非叶子节点的误差传播给隐藏层

				// Learn weights hidden -> output,更新当前非叶子节点的向量
				for (j = 0; j < dimension; j++) 
					mapVec[j + startPos] += gradient * hiddenVec[j] * senPro[0];
			}
		}

		if(negative > 0)//采用neg体系
		{
			for (i = 0; i < negative + 1; i++) 
			{
				if (i == 0) 
				{
					target = position;//中心词
					label = 1;//正样本
				}
				else 
				{
					next_random = next_random * (unsigned long long)25214903917 + 11;
					target = table[(next_random >> 16) % table_size];

					if (target == 0)//遇到了当前中心词 
						target = next_random % (vocab_size - 1) + 1;
					if (target == position) 
						continue;

					label = 0;//负样本
				}

				startPos = target * dimension;//负样本信息在mapVecNeg中的开始位置

				for (j = 0; j < dimension; j++) 
					innerProd += hiddenVec[j] * mapVecNeg[j + startPos];//计算内积，隐层不变

				if (innerProd > MAX_EXP)
					gradient = (label - 1) * alpha;
				else if (innerProd < -MAX_EXP) 
					gradient = (label - 0) * alpha;
				else 
					gradient = (label - expTable[(int)((innerProd + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;//这里的梯度，实际上是梯度和学习速率的乘积

				for (j = 0; j < dimension; j++) 
					hiddenVecE[j] += gradient * mapVecNeg[j + startPos] * senPro[0];//隐层的误差
				for (j = 0; j < dimension; j++) 
					mapVecNeg[j + startPos] += gradient * hiddenVec[j] * senPro[0];//更新负样本向量
			}
		}

		// hidden -> in
		for (i = 1; i < length; i++)//对input layer的word vector进行更新
		{
			cwPos = sen[i];//当前上下文词在词表中的位置下标

			for (j = 0; j < dimension; j++) 
				vector[j + cwPos * dimension] += hiddenVecE[j];
		}
	}
}

//传统的Skip-garm模型的驯良
void trainTraditionalSkipgram(long long *sen, long long length, real *hiddenVec, real *hiddenVecE, unsigned long long next_random)
{
	long long i,j,k;

	long long cwPos;//当前上下文词在词表中的位置下标
	long long startPos_1;//当前遍历到的上下文词的向量表示在vector中的开始位置
	long long startPos_2;//当前遍历到的非叶子节点的权重在mapVec中的开始位置

	long long target, label;

	real innerProd=0;//内积
	real gradient=0;//梯度

	for (i = 1; i < length; i++) 
	{
		cwPos=sen[i];//当前上下文词在词表中的位置信息

		startPos_1 = cwPos * dimension;//当前遍历到的上下文词的向量表示的开始位置

		if (hs)//层次softmax体系 
		{
			for (j = 0; j < vocabulary[sen[0]].codelen; j++) //逐节点遍历当前中心词存储的huffman树中的路径
			{
				innerProd = 0;//内积
				startPos_2 = vocabulary[sen[0]].point[j] * dimension;//当前中心词的遍历到的非叶子节点的权重的开始位置

				// Propagate hidden -> output
				for (k = 0; k < dimension; k++) 
					innerProd += vector[k + startPos_1] * mapVec[k + startPos_2];//计算两个词向量的内积(没有隐层) 

				if (innerProd <= -MAX_EXP) 
					continue;
				else if (innerProd >= MAX_EXP) 
					continue;
				else 
					innerProd = expTable[(int)((innerProd + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];

				gradient = (1 - vocabulary[sen[0]].code[j] - innerProd) * alpha;// 'g' is the gradient multiplied by the learning rate

				// Propagate errors output -> hidden
				for (k = 0; k < dimension; k++) 
					hiddenVecE[k] += gradient * mapVec[k + startPos_2];//隐藏层的误差

				// Learn weights hidden -> output
				for (k = 0; k < dimension; k++) 
					mapVec[k + startPos_2] += gradient * vector[k + startPos_1];//更新非叶子节点权重表示
			}
		}

		if (negative > 0)//neg体系
		{ 
			for (j = 0; j < negative + 1; j++) 
			{
				if (j == 0) 
				{
					target = sen[0];
					label = 1;
				}
				else 
				{
					next_random = next_random * (unsigned long long)25214903917 + 11;
					target = table[(next_random >> 16) % table_size];

					if (target == 0) 
						target = next_random % (vocab_size - 1) + 1;

					if (target == sen[0]) 
						continue;
					label = 0;
				}

				startPos_2 = target * dimension;

				innerProd = 0;
				for (k = 0; k < dimension; k++) 
					innerProd += vector[k + startPos_1] * mapVecNeg[k + startPos_2];

				if (innerProd > MAX_EXP) 
					gradient = (label - 1) * alpha;
				else if (innerProd < -MAX_EXP) 
					gradient = (label - 0) * alpha;
				else 
					gradient = (label - expTable[(int)((innerProd + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;

				for (k = 0; k < dimension; k++) 
					hiddenVecE[k] += gradient * mapVecNeg[k + startPos_2];
				for (k = 0; k < dimension; k++) 
					mapVecNeg[k + startPos_2] += gradient * vector[k + startPos_1];
			}
		}

		// Learn weights input -> hidden
		for (k = 0; k < dimension; k++) 
			vector[k + startPos_1] += hiddenVecE[k];//更新周围几个词语的向量
	}
}

//基于词图的Skip-garm模型的训练
void trainWordLatticeSkipgram(long long *sen, real *senPro, long long length, real *hiddenVec, real *hiddenVecE, unsigned long long next_random)
{
	long long i,j,k;

	long long cwPos;//当前上下文词在词表中的位置下标
	long long startPos_1;//当前遍历到的上下文词的向量表示在vector中的开始位置
	long long startPos_2;//当前遍历到的非叶子节点的权重在mapVec中的开始位置

	long long target, label;

	real innerProd=0;//内积
	real gradient=0;//梯度
	real sumPro=0;//上下文词的概率值和

	for(k=1;k<length;k++)//计算上下文词的概率值和，用作归一化
		sumPro+=senPro[k];

	for (i = 1; i < length; i++) 
	{
		cwPos=sen[i];//当前上下文词在词表中的位置信息

		startPos_1 = cwPos * dimension;//当前遍历到的上下文词的向量表示的开始位置

		if (hs)//层次softmax体系 
		{
			for (j = 0; j < vocabulary[sen[0]].codelen; j++) //逐节点遍历当前中心词存储的huffman树中的路径
			{
				innerProd = 0;//内积
				startPos_2 = vocabulary[sen[0]].point[j] * dimension;//当前中心词的遍历到的非叶子节点的权重的开始位置

				// Propagate hidden -> output
				for (k = 0; k < dimension; k++) 
					innerProd += vector[k + startPos_1] * mapVec[k + startPos_2];//计算两个词向量的内积(没有隐层) 

				if (innerProd <= -MAX_EXP) 
					continue;
				else if (innerProd >= MAX_EXP) 
					continue;
				else 
					innerProd = expTable[(int)((innerProd + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];

				gradient = (1 - vocabulary[sen[0]].code[j] - innerProd) * alpha;// 'g' is the gradient multiplied by the learning rate

				// Propagate errors output -> hidden
				for (k = 0; k < dimension; k++) 
					hiddenVecE[k] += gradient * mapVec[k + startPos_2] * (senPro[0]*senPro[i]/sumPro);//隐藏层的误差

				// Learn weights hidden -> output
				for (k = 0; k < dimension; k++) 
					mapVec[k + startPos_2] += gradient * vector[k + startPos_1] * (senPro[0]*senPro[i]/sumPro);//更新非叶子节点权重表示
			}
		}

		if (negative > 0)//neg体系
		{ 
			for (j = 0; j < negative + 1; j++) 
			{
				if (j == 0) 
				{
					target = sen[0];
					label = 1;
				}
				else 
				{
					next_random = next_random * (unsigned long long)25214903917 + 11;
					target = table[(next_random >> 16) % table_size];

					if (target == 0) 
						target = next_random % (vocab_size - 1) + 1;

					if (target == sen[0]) 
						continue;
					label = 0;
				}

				startPos_2 = target * dimension;

				innerProd = 0;
				for (k = 0; k < dimension; k++) 
					innerProd += vector[k + startPos_1] * mapVecNeg[k + startPos_2];

				if (innerProd > MAX_EXP) 
					gradient = (label - 1) * alpha;
				else if (innerProd < -MAX_EXP) 
					gradient = (label - 0) * alpha;
				else 
					gradient = (label - expTable[(int)((innerProd + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;

				for (k = 0; k < dimension; k++) 
					hiddenVecE[k] += gradient * mapVecNeg[k + startPos_2] * (senPro[0]*senPro[i]/sumPro);
				for (k = 0; k < dimension; k++) 
					mapVecNeg[k + startPos_2] += gradient * vector[k + startPos_1] * (senPro[0]*senPro[i]/sumPro);
			}
		}

		// Learn weights input -> hidden
		for (k = 0; k < dimension; k++) 
			vector[k + startPos_1] += hiddenVecE[k];//更新周围几个词语的向量
	}
}

//模型训练线程，在执行线程之前要根据词频排序的词表对每个词求得huffman编码
void *trainModelThread(void *id) 
{
	long long i=0;
	char ch;

	char content[MAX_STRING];//保存构造句子时从文件中读出的内容
	bool begin=true;//表示需要跳过不完整的句子
	int flag=1;//判断当前获得的信息是词信息还是频次信息

	long long word;//向句中添加单词用，句子完成后表示句子中的当前单词(即中心词)在词表中的位置
	long long sentence_length=0;//当前句子的长度 
	long long pro_length=0;//当前句子中对应词的概率值所在下标
	
	long long word_count = 0;//已训练词总个数
	long long last_word_count = 0;//保存值，以便在训练语料长度超过某个值时输出信息
	long long sen[MAX_SENTENCE_LENGTH + 1];//当前句子
	real senPro[MAX_SENTENCE_LENGTH + 1];//当前句子中每个词对应的概率值
	
	long long local_iter = iter;
	unsigned long long next_random = (long long)id;
	clock_t now;
	
	real *hiddenVec = (real *)calloc(dimension, sizeof(real));//隐层结点
	real *hiddenVecE = (real *)calloc(dimension, sizeof(real));//误差累计项
	FILE *fi = fopen(trainFile, "rb");
	
	 //每个线程对应一段文本。根据线程id找到自己负责的文本的初始位置 
	fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
	do ch = fgetc(fin);
	while (ch != '\n')//读入当前字符
	
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
			flag = 1;
			while (1) 
			{
				readWordForSen(content, fi);//从文件中读取单词来构造当前句子

				if (feof(fi)) 
					break;

				if(flag==1)//当前从文件中读出的内容是词
				{
					word = searchVocab(content);
					if (word == -1)
						continue;

					word_count++;
					if (word == 0)//读到了句子结束标识
						break;

					//下采样过程中会随机丢弃高频词的词，但要保持排序的一致 ,当前测试禁用下采样
					if (false) // (sample > 0) 
					{
						real ran = (sqrt(vocabulary[word].cn / (sample * train_words)) + 1) * 
							(sample * train_words) / vocabulary[word].cn;
						next_random = next_random * (unsigned long long)25214903917 + 11;
						if (ran < (next_random & 0xFFFF) / (real)65536)
							continue;
					}

					sen[sentence_length] = word;
					sentence_length++;
					if (sentence_length >= MAX_SENTENCE_LENGTH) 
						break;
				}
				else
				{
					senPro[pro_length] = atof(content); //保存当前词的概率值
					//memcpy(&senPro[pro_length], content, 20);
					pro_length++;
					assert(pro_length == sentence_length); //debug
				}	
				flag = !flag;
			}
		}//句子读入结束
		
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
			do ch = fgetc(fin);
			while (ch != '\n')//读入当前字符
			
			continue;//开始新的一轮迭代
		}
		
		word = sen[0];//中心词在词表中的位置 
		if (word == -1) 
			continue;
		
		for (i = 0; i < dimension; i++) //初始化隐层单元
			hiddenVec[i] = 0;
		for (i = 0; i < dimension; i++) //初始化误差累计项
			hiddenVecE[i] = 0;
		
		//训练CBOW模型
		trainTraditionalCBOW(sen,sentence_length,hiddenVec,hiddenVecE,next_random,word);//传统CBOW模型的训练(hs+neg)
		trainWordLatticeCBOW(sen,senPro,sentence_length,hiddenVec,hiddenVecE,next_random,word);//基于词图的CBOW模型的训练(hs+neg)
		
		//训练skip-gram模型
		trainTraditionalSkipgram(sen,sentence_length,hiddenVec,hiddenVecE,next_random);//传统ship-gram模型的训练(hs+neg)
		trainWordLatticeSkipgram(sen,senPro,sentence_length,hiddenVec,hiddenVecE,next_random);//基于词图的ship-gram模型的训练(hs+neg)

		//清空当前句子长度信息
		sentence_length=0;
	
	}//while结束
	
	fclose(fi);
	free(hiddenVec);
	free(hiddenVecE);
	pthread_exit(NULL);
}

//模型训练
void trainModel() 
{
	long i,j;//迭代变量
	FILE *fo;
	
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	printf("Starting training using file %s\n", trainFile);
	
	starting_alpha = alpha;//保存学习速率的初始值
	if (vocabReadFile[0] != 0) 
		readVocab(); //从文件读入词表
	else
		learnVocabFromTrainFile();//从训练文件学习词汇  
	
	if (vocabSaveFile[0] != 0) 
		saveVocab();//保存词表
	if (outputFile[0] == 0) 
		return;
	
	initNet();//网络初始化
	if (negative > 0) 
		initUnigramTable();//构建负采样算法中的权值表
	
	start = clock();//获取训练的开始时间
	for (i = 0; i < num_threads; i++) //线程创建
		pthread_create(&pt[i], NULL, trainModelThread, (void *)i);
		
	//以阻塞的方式等待指定的线程结束。当函数返回时，被等待线程的资源被收回。如果线程已经结束，那么该函数会立即返回。并且指定的线程必须是joinable的。	
	for (i = 0; i < num_threads; i++) 
		pthread_join(pt[i], NULL);
	
	//在这里，所有的线程都已经执行结束了
	
	fo = fopen(outputFile, "wb");

	// Save the word vectors
	fprintf(fo, "%lld %lld\n", vocab_size, dimension);

	for (i = 0; i < vocab_size; i++) //遍历词表
	{
		fprintf(fo, "%s ", vocabulary[i].word);
		for (j = 0; j < dimension; j++) //写入当前词的每一维
			fprintf(fo, "%lf ", vector[i * dimension + j]);				

		fprintf(fo, "\n");
	}
	fclose(fo);
}

//用于变量的控件分配和expTable的计算
void init()
{
	int i;//迭代变量

	vocabulary = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));//为词表分配空间
	hashTable = (int *)calloc(vocab_hash_size, sizeof(int));//为hash表分配空间
	expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));//为存储sigmoid函数计算结果的表分配空间

	//计算预定义范围内的sigmoid函数值
	for (i = 0; i < EXP_TABLE_SIZE; i++) 
	{
		expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
		expTable[i] = expTable[i] / (expTable[i] + 1); // Precompute f(x) = x / (x + 1)
	}
}

//用于读取命令行参数,其中argc为argument count，argv为argument value
int argPos(char *str, int argc, char **argv) 
{
	int i;//遍历命令行参数
	for (i= 1; i < argc; i++) //依次读取参数,跳过可执行文件名所在的位置
	{
		if (!strcmp(str, argv[i])) 
		{
			if (i == argc - 1) 
			{
				printf("Argument missing for %s\n", str);
			    exit(1);
			}
			
			return i;
		}
	}
	return -1;
}

//主函数
/*完成的内容包括：接收参数，分配空间，计算sigmoid函数值表*/
int main(int argc, char **argv) 
{
	int i=0;//遍历命令行参数

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
		
		printf("\t-debug <int>\n");//debug模式，默认是2，表示在训练过程中会输出更多信息 
		printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
		
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
	
	outputFile[0] = 0;
	vocabSaveFile[0] = 0;
	vocabReadFile[0] = 0;

	if ((i = argPos((char *)"-size", argc, argv)) > 0) dimension = atoi(argv[i + 1]);//atoi是字符串转整型的函数
	if ((i = argPos((char *)"-train", argc, argv)) > 0) strcpy(trainFile, argv[i + 1]);
	if ((i = argPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(vocabSaveFile, argv[i + 1]);
	if ((i = argPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(vocabReadFile, argv[i + 1]);
	if ((i = argPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
	if ((i = argPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
	if (cbow) alpha = 0.05;//修改采用cbow模式时所采用的默认学习速率
	if ((i = argPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
	if ((i = argPos((char *)"-output", argc, argv)) > 0) strcpy(outputFile, argv[i + 1]);
	if ((i = argPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
	if ((i = argPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
	if ((i = argPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
	if ((i = argPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
	if ((i = argPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
	if ((i = argPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
	
	init();//初始化
	trainModel();//模型训练

	return 0;
}