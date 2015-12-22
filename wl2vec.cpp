#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100 //�ʵ���󳤶�
#define EXP_TABLE_SIZE 1000 //expTable�ܴ洢�Ľ������
#define MAX_EXP 6 //����sigmoid����ֵʱ��e��ָ�������6����С��-6
#define MAX_SENTENCE_LENGTH 1000 //���ľ��ӳ��ȣ�ָ���дʵĸ���
#define MAX_CODE_LENGTH 40 //���huffman���볤��

const int vocab_hash_size = 30000000;//��ϣ��, Maximum 30 * 0.7 = 21M words in the vocabulary
typedef float real;//ʵ������

//�ʱ��е����ʵ����ݽṹ
struct vocab_word 
{
	long long cn; //��Ƶ
	long long procn;//��Ȩ��Ƶ
	int *point; //huffman���дӸ��ڵ㵽��ǰ������Ӧ��Ҷ�ӽڵ��·���У��������з�Ҷ�ӽڵ������
	char *word, *code, codelen; //��ǰ�ʣ����Ӧ��huffman���뼰���볤��
};

char trainFile[MAX_STRING], outputFile[MAX_STRING]; 
char vocabSaveFile[MAX_STRING], vocabReadFile[MAX_STRING]; 

struct vocab_word *vocabulary;//�ʱ�

//���е�min_reduce��ָ����С�ʱ�����л�ɾ����ƵС�����ֵ�Ĵ�
int cbow = 1, debug_mode = 2, min_count = 5, num_threads = 12, min_reduce = 1;

int *hashTable;//�ʻ���hash�洢

//�ֱ�Ϊ�ʱ�Ĵ�С���ʱ�ĵ�ǰʵ�ʴ�С��input layer��ÿ��word vector��ά��
long long vocab_max_size = 1000, vocab_size = 0, dimension = 100;

//�ֱ�Ϊ��ѵ���Ĵ�����,��ѵ���Ĵʸ���������������ѵ���ļ��Ĵ�С
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0;

real alpha = 0.025, starting_alpha, sample = 1e-3;

/*�ֱ�Ϊinput layer�Ĵ��������У�softmax��hidden layer��huffman tree�з�Ҷ�ӽڵ��ӳ��Ȩ�أ�negative sampling��hidden layer��output layer��ӳ��Ȩ�أ�����sigmoid����ֵ�ı�*/
real *vector, *mapVec, *mapVecNeg, *expTable;

clock_t start;//����ʼʱ��

int hs = 0, negative = 5;//ǰ�ߣ��Ƿ����softmax��ϵ�ı�ʶ�����ߣ�������������
const int table_size = 1e8;//������Ĵ�С����negative sampling�л��õ�
int *table;//������

//����һ���ʵ�hashֵ
int getWordHash(char *word) 
{
	unsigned long long i;//��������
	unsigned long long hash = 0;
	
	for (i = 0; i < strlen(word); i++) //����word�е�ÿ���ַ�������257���ƹ��쵱ǰword��key value
		hash = hash * 257 + word[i];
	
	hash = hash % vocab_hash_size;//����������
	
	return hash;
}


//�Ƴ���Ƶ��С�Ĵʣ������ʱ�
void reduceVocab() 
{
	int i, j = 0;
	unsigned int hash;

	for (i = 0; i < vocab_size; i++) //�����ʱ�
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
	
	vocab_size = j;//��ǰ�ʱ��������������ʵ�ʴ�С
	for (i = 0; i < vocab_hash_size; i++)
		hashTable[i] = -1;
	for (i = 0; i < vocab_size; i++)//���¹�������������Ĵʱ��hash��
	{
		hash = getWordHash(vocabulary[i].word);
		while (hashTable[hash] != -1)
			hash = (hash + 1) % vocab_hash_size;
		hashTable[hash] = i;
	}
	
	fflush(stdout);
	min_reduce++;//ע��������Ƶ����ֵ
}

//���ص�ǰ���ڴʱ��е�λ�ã���������ھͷ���-1
int searchVocab(char *word) 
{
	unsigned int hash = getWordHash(word);//��õ�ǰ�ʵ�hashֵ
	
	while (1) 
	{
		if (hashTable[hash] == -1)//��ǰhashֵ��hash���в����ڣ�����ǰ�ʲ����ڣ�����-1
			return -1;
			
		//�õ�ǰ����hashֵ�ڴʱ����ҵ��Ĵ��뵱ǰ�ʶԱȣ������ͬ�����ʾ��ǰ��hashֵ��Ӧ��ȷ
		if (!strcmp(word, vocabulary[hashTable[hash]].word)) 
			return hashTable[hash];
		
		hash = (hash + 1) % vocab_hash_size;//���Ŷ�ַ��
	}
	return -1;
}

//�Ƚ������ʴ�Ƶ�Ĵ�С
int vocabCompare(const void *a, const void *b) 
{
	return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

//���ʱ��մ�Ƶ��������
void sortVocab() 
{
	int i, size;
	unsigned int hash;
	
	// Sort the vocabulary and keep </s> at the first position
	qsort(&vocabulary[1], vocab_size - 1, sizeof(struct vocab_word), vocabCompare);
	
	for (i = 0; i < vocab_hash_size; i++) //���³�ʼ��hash��
		hashTable[i] = -1;
	
	size = vocab_size;//������ǰ�ʱ��ԭʼ��С
	train_words = 0;//��ѵ���ĴʵĴ�Ƶ�Ӻ�ֵ��ʱΪ0
	
	for (i = 0; i < size; i++) //������ǰ�ʱ����е����д�
	{
		//�жϵ�ǰ�ʵĴ�Ƶ�Ƿ�С����Сֵ���ǵĻ��ʹӴʱ���ȥ��
		if ((vocabulary[i].cn < min_count) && (i != 0)) 
		{
			vocab_size--;
			free(vocabulary[i].word);
		} 
		else 
		{
			// �������Ҫ���¼���hash����
			hash=getWordHash(vocabulary[i].word);
			while (hashTable[hash] != -1)
				hash = (hash + 1) % vocab_hash_size;//���ŵ�ַ�������ͻ
			hashTable[hash] = i;
			
			train_words += vocabulary[i].cn;
		}
	}
	
	vocabulary = (struct vocab_word *)realloc(vocabulary, (vocab_size + 1) * sizeof(struct vocab_word));//׷�ӿռ�
	
	//Ϊ�������ṹ����ռ�
	for (i = 0; i < vocab_size; i++) 
	{
		vocabulary[i].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
		vocabulary[i].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
	}
}

//��һ������ӵ��ʱ���  
int addWordToVocab(char *word) 
{
	unsigned int hash, length = strlen(word) + 1;//��ǰ�ʵ�hashֵ��length
	
	if (length > MAX_STRING) 
		length = MAX_STRING;
	
	vocabulary[vocab_size].word = (char *)calloc(length, sizeof(char));//Ϊ������Ĵʵ�λ�÷���ռ�
	
	strcpy(vocabulary[vocab_size].word, word);
	vocabulary[vocab_size].cn = 0;//�����Ƶ��Ϊ0
	vocab_size++;
	
	// Reallocate memory if needed
	if (vocab_size + 2 >= vocab_max_size) 
	{
		vocab_max_size += 1000;
		vocabulary = (struct vocab_word *)realloc(vocabulary, vocab_max_size * sizeof(struct vocab_word));
	}
	
	hash = getWordHash(word);//��ȡ��ǰ�¼���Ĵʵ�hashֵ
	while (hashTable[hash] != -1)
		hash = (hash + 1) % vocab_hash_size;//ʹ�ÿ��ŵ�ַ�������ͻ
	
	//�ڹ�ϣ���б��浱ǰ���ڴʱ��е�λ��
	hashTable[hash] = vocab_size - 1;
	
	return vocab_size - 1;//�����¼���Ĵ��ڴʱ��е��±�
}

//���ļ��ж�ȡ�����Ĵ�
void readWord(char *word, FILE *fin) 
{
	int index = 0;//������ǰ��ȡ�����ַ���word�е�λ��
	int ch;//���浱ǰ��ȡ�����ַ�

	while (!feof(fin))//�ж��Ƿ񵽴��ļ���β 
	{
		ch = fgetc(fin);//���뵥��

		if (ch == '\n') //��Ҫ������һ��
			continue;

		if ((ch == ' ') || (ch == '\t') || (ch == '\n'))//�������ʱ߽� 
		{
			if (index > 0) //word���Ѿ�����������
			{
				if (ch == '\n') 
					ungetc(ch, fin);//��һ���ַ��˻ص��������У�����˻ص��ַ�������һ����ȡ�ļ����ĺ���ȡ��
				break;
			}

			if (ch == '\n') 
				continue;
		}

		word[index] = ch;
		index++;

		if (index >= MAX_STRING - 1)//�ضϹ����ĵ���
			index--;
	}

	word[index] = 0;//���Ͻ�������\0��
}

//��ѵ���ļ��ж�ȡһ����(���ڹ����ʱ�)���ո�Ϊ���ʱ߽�
void readWordFromFile(char *word,FILE *fin) 
{
	int index=0;//����word���ַ���λ��
	char ch;//���浱ǰ��ȡ�����ַ�

	while (!feof(fin))//�ж��Ƿ񵽴��ļ���β 
	{
		ch = fgetc(fin);//���뵱ǰ�ַ�
		while(ch!=' ')//��������Ϊ���ʱ߽�Ŀո�֮ǰ���ǵ�ǰ��
		{
			if(ch=='*')
			{
				ch = fgetc(fin);
				break;
			}

			word[index]=ch;
			index++;

			if (index >= MAX_STRING - 1)//�ضϹ����ĵ���
				index--;

			ch = fgetc(fin);
		}

		word[index] = 0;//���Ͻ�������\0��
		/*if(index==0)//��ǰword��û�б�����Ч��
		{
			ch = fgetc(fin);
			break;
		}*/

		//��ǰ�����˵��ʱ߽磬��������֪�������س���ʾ����
		while(ch!='\n')
			ch = fgetc(fin);

		//��ʱ�ļ�ָ�뵽����һ��
		break;
	}
}

//��ȡ�ʱ�
void readVocab() 
{
	long long i = 0;
	char c;//���ڶ�ȡ��Ƶ
	char word[MAX_STRING];//���ڱ��浱ǰ���ļ��ж����Ĵ�
	
	FILE *fin = fopen(vocabReadFile, "rb");//�򿪴ʱ��ļ�
	if (fin == NULL) 
	{
		printf("Vocabulary file not found\n");
		exit(1);
	}
	
	//hash���ʼ��
	for (i = 0; i < vocab_hash_size; i++) 
		hashTable[i] = -1;
	
	vocab_size = 0;//��ǰ�ʱ��ʵ�ʴ�СΪ0
	while (1) 
	{
		readWord(word, fin);//���ļ��ж���һ����
		if (feof(fin)) 
			break;
		
		i = addWordToVocab(word);//����ǰ���ļ��ж�ȡ���Ĵʼ��뵽�ʱ��У��������ڴʱ��е�λ��
		fscanf(fin, "%lld%c", &vocabulary[i].cn, &c);//��ȡ��Ƶ
		
		vocab_size++;
	}
	
	sortVocab();//���ݴ�Ƶ����
	
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
	file_size = ftell(fin);//��õ�ǰѵ�������ļ��Ĵ�С
	fclose(fin);
}

//��ѵ���ļ��ж�ȡ����������ʱ�
void learnVocabFromTrainFile()
{
	char word[MAX_STRING];//��ѵ���ļ��ж����Ĵ�
	FILE *fin;
	long long i, index;
	
	for (i = 0; i < vocab_hash_size; i++) //hash���ʼ��
		hashTable[i] = -1;
	
	fin = fopen(trainFile, "rb");//���ļ�
	if (fin == NULL) 
	{
		printf("ERROR: training data file not found!\n");
		exit(1);
	}
	
	vocab_size = 0;//�ʱ�ǰʵ�ʴ�СΪ0
	addWordToVocab((char *)"</s>");//��֤</s>����ǰ��
	
	while (1) 
	{
		readWordFromFile(word, fin);//��ѵ���ļ��ж�ȡһ����
		if (feof(fin)) 
			break;
		train_words++;//��ѵ�����ʸ�����1
		
		if ((debug_mode > 1) && (train_words % 100000 == 0)) 
		{
			printf("%lldK%c", train_words / 1000, 13);
			fflush(stdout);
		}
		
		index = searchVocab(word);//���ظô��ڴʱ��е�λ��
		if (index == -1)//��ʱ�ôʲ������ڴʱ�֮�У��ͽ������ʱ�֮��  
		{
			i = addWordToVocab(word);
			vocabulary[i].cn = 1;
		} 
		else 
			vocabulary[index].cn++;//���´�Ƶ
		
		if (vocab_size > vocab_hash_size * 0.7) //����ʱ�̫�Ӵ󣬾������ʱ�
			reduceVocab();
	}
	
	sortVocab();//���ݴ�Ƶ���ʱ�����
	
	if (debug_mode > 0) 
	{
		printf("Vocab size: %lld\n", vocab_size);
		printf("Words in train file: %lld\n", train_words);
	}
	
	file_size = ftell(fin);//���ѵ���ļ��Ĵ�С
	fclose(fin);
}

//����ʱ�
void saveVocab() 
{
	long long i;
	FILE *fo = fopen(vocabSaveFile, "wb");
	for (i = 0; i < vocab_size; i++) 
		fprintf(fo, "%s %lld\n", vocabulary[i].word, vocabulary[i].cn);
	fclose(fo);
}

//ͨ����Ƶ��Ϣ����huffman������Ƶ�ߵĴʵ�huffman��������
void createBinaryTree() 
{
	long long i,j;
	long long index;
	long long position_1,position_2;//�ڹ���huffman���Ĺ�������ѡ��Ȩ����С�Ľڵ��ڴʱ��е��±�
	long long min_1,min_2;//�����������ҵ�����СȨ��

	long long point[MAX_CODE_LENGTH];//�Ӹ��ڵ㵽��ǰ�ʵ�·��
	char code[MAX_CODE_LENGTH];//��ǰ�ʵ�huffman code
	
	//�����ڹ���huffman���������õ��Ľڵ��Ƶ��Ϣ
	long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));

	//��������֧�ı�����Ϣ����������С�ߵı���Ϊ0����С�ߵı���Ϊ1
	long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));

	//���游�׽ڵ��ڴʱ��е�λ����Ϣ
	long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	
	for (i = 0; i < vocab_size; i++) //��ʼ����ǰҶ�ӽڵ��Ƶ��Ϣ(��Ȩ��)
		count[i] = vocabulary[i].cn;
	for (i = vocab_size; i < vocab_size * 2; i++) //����δ֪�ķ�Ҷ�ӽڵ��Ȩ�أ�ȫ����һ��������ʼ��
		count[i] = 1e15;
	
	position_1 = vocab_size - 1;
	position_2 = vocab_size;
	
	//����huffman��
	for (i = 0; i < vocab_size - 1; i++)//�����ʱ� 
	{
		// �ҳ�ĿǰȨֵ��С�������ڵ� 
		if (position_1 >= 0) //��һ��Ȩֵ��С�Ľڵ�  
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
		
		if (position_1 >= 0)//�ڶ���Ȩֵ��С�Ľڵ�  
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
		
		count[vocab_size + i] = count[min_1] + count[min_2];//���¹���ķ�Ҷ�ӽڵ��Ȩ�ؼ��뵽Ȩ�ؼ���
		
		//Ϊ���ӽڵ㱣�浱ǰ���׽ڵ��ڴʱ��е�λ��
		parent_node[min_1] = vocab_size + i;
		parent_node[min_2] = vocab_size + i;
		
		binary[min_2] = 1;//�ڵ����Ϊ1��֮ǰĬ����0 
	}
	
	//������õ�huffman�������ʱ��е�ÿ����
	for (i = 0; i < vocab_size; i++) 
	{
		j = i;//���ڱ������Ĵ��ڴʱ��е�λ���±�
		index = 0;//��¼���ݵķ�֧����
		
		while (1) 
		{
			code[index] = binary[j];
			point[index] = j;
			index++;

			if(index>MAX_CODE_LENGTH)//��ǰ��������ȳ����涨�������Ⱦ��˳�
			{
				printf("Current index has exceeded the max_code_length!\n");
				exit(1);
			}
			
			j = parent_node[j];//��������ڵ����
			if (j == vocab_size * 2 - 2) //�Ѿ�������ڵ�
				break; 
		}
		
		vocabulary[i].codelen = index;
		vocabulary[i].point[0] = vocab_size - 2;//�����ڵ����·����
		for (j = 0; j < index; j++) 
		{
			vocabulary[i].code[index - j - 1] = code[j];
			vocabulary[i].point[index - j] = point[j] - vocab_size;
		}
	}
	
	//�ͷſռ�
	free(count);
	free(binary);
	free(parent_node);
}

//����ģ�ͳ�ʼ��,��������huffman��
void initNet() 
{
	long long i, j;
	unsigned long long next_random = 1;
	
	posix_memalign((void **)&vector, 128, (long long)vocab_size * dimension * sizeof(real));//Ԥ�����ڴ�ķ���
	if (vector == NULL) //��̬�ڴ����ʧ��
	{
		printf("Memory allocation failed\n"); 
		exit(1);
	}
	
	if (hs) //����softmax��ϵ
	{
		//Ϊhuffman tree�з�Ҷ�ӽڵ��Ȩ��ӳ����䶯̬�ڴ�
		posix_memalign((void **)&mapVec, 128, (long long)vocab_size * dimension * sizeof(real));
		if (mapVec == NULL) 
		{
			printf("Memory allocation failed\n"); 
			exit(1);
		}
		
		for (i = 0; i < vocab_size; i++) 
		{
			for (j = 0; j < dimension; j++)
				mapVec[i * dimension + j] = 0;//ÿ����Ҷ�ӽڵ��Ȩ����mapVec����dimension����Ԫ,˳����
		}
	}
	
	if (negative>0) //negative sampleing��ϵ
	{
		posix_memalign((void **)&mapVecNeg, 128, (long long)vocab_size * dimension * sizeof(real));//Ԥ�����ڴ����
		if (mapVecNeg == NULL) 
		{
			printf("Memory allocation failed\n");
			exit(1);
		}
		
		for (i = 0; i < vocab_size; i++) 
		{
			for (j = 0; j < dimension; j++)
				mapVecNeg[i * dimension + j] = 0;//ÿ���������ı�ʾ��mapVecNeg����dimension����Ԫ,˳����
		}
	}
	
	for (i = 0; i < vocab_size; i++) //�����ʼ��input layer�е�ÿ��word vector
	{
		for (j = 0; j < dimension; j++) 
		{
		next_random = next_random * (unsigned long long)25214903917 + 11;
		vector[i * dimension + j] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / dimension;
		}
	}
	
	createBinaryTree();//����huffman�����Դʱ��е�ÿ���ʽ��б���
}

//�����������㷨�е�Ȩֵ�ֲ���
void initUnigramTable() 
{
	int i, j;//ѭ������

	double train_words_pow = 0;//�ʵ�Ȩ����ֵ����������һ������
	double len, power = 0.75;//��ÿ���ʶ���Ȩֵ��ʱ����ֱ��ȡcount�����Ǽ����˦����ݣ��˴���ȡ��0.75

	table = (int *)malloc(table_size * sizeof(int));//Ϊtable����ռ�

	for (i = 0; i < vocab_size; i++) //�����ʻ������train_words_pow
		train_words_pow += pow(vocabulary[i].cn, power);

	j = 0;//�����ʱ�
	len = pow(vocabulary[j].cn, power) / train_words_pow;//��ʼ���ѱ����Ĵʵ�ȨֵռȨ����ֵ�ı�������ʼֵΪ��һ���ʵ�Ȩֵ�������е�ռ��

	//����table�����Ⱦ��ʷ�ͶӰ���Դʱ��дʵ�Ȩ��ֵΪ��׼���еķǵȾ��ʷ���
	for (i = 0; i < table_size; i++)
	{
		table[i] = j;//��ǰtable�еĵȾ����ʷֶ�i��Ӧ���ǷǵȾ��ʷֶ�j

		if (i / (double)table_size > len) 
		{
			j++;//����ʱ��е���һ��λ��
			len += pow(vocabulary[j].cn, power) / train_words_pow;//����ǰ�ʵ�ȨֵռȨ����ֵ�ı������뵽���ۼƴʵĵ�ǰ��ֵ��
		}

		if (j >= vocab_size) //�ж��Ƿ񳬹��ʻ��Ĵ�С
			j = vocab_size - 1;
	}
}

//��ѵ��ģ��ʱΪ����һ�����Ӷ����ļ��ж���ʻ����Ӧ�ĸ���
//���߳̿�ʼ���Ա�֤���뽫����������
void readWordForSen(char *content, FILE *fin)
{
	int index = 0;//����word���ַ���λ��
	char ch;//���浱ǰ��ȡ�����ַ�

	while (!feof(fin))//�ж��Ƿ񵽴��ļ���β 
	{
		ch = fgetc(fin);//���뵱ǰ�ַ�
		
		if(feof(fin))
			break;
		/*
		if (ch == '*') //��Ӧ�ļ�β
		{
			ch = fgetc(fin);
			break;
		}*/

		if(ch=='\n')//����'\n',�򽫽����µ�һ���ٺ�������
			continue;

		/*
		if(begin)
		{
			while(ch!='\n')//Ϊ��ֹ�������ľ��ӱ�ʹ�ã�һ����������һ�����з�֮���ٽ����¾��ӵĶ���
				ch = fgetc(fin);//���뵱ǰ�ַ�
		}*/

		if ((ch == ' ') || (ch == '\t') || (ch == '\n'))//�����߽� 
		{
			if (index > 0) //word���Ѿ�����������
			{
				if (ch == '\n') 
					ungetc(ch, fin);//��һ���ַ��˻ص��������У�����˻ص��ַ�������һ����ȡ�ļ����ĺ���ȡ��
				break;
			}

			if (ch == '\n') 
			{
				strcpy(content, (char *)"</s>");//���sentence�����ı��
				return;
			} 
			else 
				continue;
		}

		content[index] = ch;
		index++;

		if (index >= MAX_STRING - 1)//�ضϹ����ĵ���
			index--;
	}

	content[index] = 0;//���Ͻ�������\0��
}

//��ͳCBOWģ�͵�ѵ��
void trainTraditionalCBOW(long long *sen, long long length, real *hiddenVec, real *hiddenVecE, unsigned long long next_random,long long position)
{
	long long i,j;

	int cw=0;//��ǰ���Ĵʵ������Ĵʸ���
	
	long long cwPos=0;//��ǰ�����Ĵ��ڴʱ��е�λ���±�
	long long startPos=0;//��ǰ�������ķ�Ҷ�ӽڵ��Ȩ����mapVec�еĿ�ʼλ��

	long long target, label;

	real innerProd=0;//�ڻ�
	real gradient=0;//�ݶ�
	
	//��input layer�� hidden layer ��ӳ��
	for ( i= 1; i < length; i++) //ɨ�赱ǰ���Ĵʵ������Ĵ�
	{
		cwPos = sen[i];//��ǰ�����Ĵ��ڴʱ��е�λ���±�

		for (j = 0; j < dimension; j++) //����hidden layer�����ݣ���input layer��
			hiddenVec[j] += vector[j + cwPos * dimension];

		cw++;
	}

	for (i = 0; i < dimension; i++) //�����㵥Ԫ��ƽ������
		hiddenVec[i] /= cw;

	if(cw)
	{
		if (hs)//���ò��softmax��ϵ 
		{
			for (i = 0; i < vocabulary[position].codelen; i++) //��ڵ������ǰ���Ĵʴ洢��huffman���е�·��
			{
				innerProd = 0;//�ڻ�
				startPos = vocabulary[position].point[i] * dimension;//��ǰ�������ķ�Ҷ�ӽڵ��Ȩ����mapVec�еĿ�ʼλ��

				// Propagate hidden -> output
				for (j = 0; j < dimension; j++) 
					innerProd += hiddenVec[j] * mapVec[j + startPos];//�����ڻ�

				if (innerProd <= -MAX_EXP) //�ڻ����ڷ�Χ����ֱ�Ӷ��� 
					continue;
				else if (innerProd >= MAX_EXP) 
					continue;
				else 
					innerProd = expTable[(int)((innerProd + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];//�ڻ�֮��sigmoid����

				gradient = (1 - vocabulary[position].code[i] - innerProd) * alpha;// ������ݶȣ�ʵ�������ݶȺ�ѧϰ���ʵĳ˻�

				// Propagate errors output -> hidden
				for (j = 0; j < dimension; j++) //���򴫲�����huffman���������ز�
					hiddenVecE[j] += gradient * mapVec[j + startPos];//�ѵ�ǰ��Ҷ�ӽڵ�����������ز�

				// Learn weights hidden -> output,���µ�ǰ��Ҷ�ӽڵ������
				for (j = 0; j < dimension; j++) 
					mapVec[j + startPos] += gradient * hiddenVec[j];
			}
		}

		if(negative > 0)//����neg��ϵ
		{
			for (i = 0; i < negative + 1; i++) 
			{
				if (i == 0) 
				{
					target = position;//���Ĵ�
					label = 1;//������
				}
				else 
				{
					next_random = next_random * (unsigned long long)25214903917 + 11;
					target = table[(next_random >> 16) % table_size];

					if (target == 0) 
						target = next_random % (vocab_size - 1) + 1;
					if (target == position) 
						continue;

					label = 0;//������
				}

				startPos = target * dimension;//��������Ϣ��mapVecNeg�еĿ�ʼλ��

				for (j = 0; j < dimension; j++) 
					innerProd += hiddenVec[j] * mapVecNeg[j + startPos];//�����ڻ������㲻��

				if (innerProd > MAX_EXP)
					gradient = (label - 1) * alpha;
				else if (innerProd < -MAX_EXP) 
					gradient = (label - 0) * alpha;
				else 
					gradient = (label - expTable[(int)((innerProd + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;//������ݶȣ�ʵ�������ݶȺ�ѧϰ���ʵĳ˻�

				for (j = 0; j < dimension; j++) 
					hiddenVecE[j] += gradient * mapVecNeg[j + startPos];//��������
				for (j = 0; j < dimension; j++) 
					mapVecNeg[j + startPos] += gradient * hiddenVec[j];//���¸���������
			}
		}

		// hidden -> in
		for (i = 1; i < length; i++)//��input layer��word vector���и���
		{
			cwPos = sen[i];//��ǰ�����Ĵ��ڴʱ��е�λ���±�

			for (j = 0; j < dimension; j++) 
				vector[j + cwPos * dimension] += hiddenVecE[j];
		}
	}
}

//���ڴ�ͼ��CBOWģ�͵�ѵ��
void trainWordLatticeCBOW(long long *sen, real *senPro, long long length, real *hiddenVec, real *hiddenVecE, unsigned long long next_random,long long position)
{
	long long i,j;

	int cw=0;//��ǰ���Ĵʵ������Ĵʸ���
	
	long long cwPos=0;//��ǰ�����Ĵ��ڴʱ��е�λ���±�
	long long startPos=0;//��ǰ�������ķ�Ҷ�ӽڵ��Ȩ����mapVec�еĿ�ʼλ��

	long long target, label;

	real innerProd=0;//�ڻ�
	real gradient=0;//�ݶ�

	//��input layer�� hidden layer ��ӳ��
	for ( i= 1; i < length; i++) //ɨ�赱ǰ���Ĵʵ������Ĵ�
	{
		cwPos = sen[i];//��ǰ�����Ĵ��ڴʱ��е�λ���±�

		for (j = 0; j < dimension; j++) //����hidden layer�����ݣ���input layer��
			hiddenVec[j] += vector[j + cwPos * dimension] * senPro[cwPos];

		cw++;
	}

	for (i = 0; i < dimension; i++) //�����㵥Ԫ��ƽ������
		hiddenVec[i] /= cw;

	if(cw)
	{
		if (hs)//���ò��softmax��ϵ 
		{
			for (i = 0; i < vocabulary[position].codelen; i++) //��ڵ������ǰ���Ĵʴ洢��huffman���е�·��
			{
				startPos = vocabulary[position].point[i] * dimension;//��ǰ�������ķ�Ҷ�ӽڵ��Ȩ����mapVec�еĿ�ʼλ��

				// Propagate hidden -> output
				for (j = 0; j < dimension; j++) 
					innerProd += hiddenVec[j] * mapVec[j + startPos];//�����ڻ�

				if (innerProd <= -MAX_EXP) //�ڻ����ڷ�Χ����ֱ�Ӷ��� 
					continue;
				else if (innerProd >= MAX_EXP) 
					continue;
				else 
					innerProd = expTable[(int)((innerProd + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];//�ڻ�֮��sigmoid����

				gradient = (1 - vocabulary[position].code[i] - innerProd) * alpha;// ������ݶȣ�ʵ�������ݶȺ�ѧϰ���ʵĳ˻�

				// Propagate errors output -> hidden
				for (j = 0; j < dimension; j++) //���򴫲�����huffman���������ز�
					hiddenVecE[j] += gradient * mapVec[j + startPos] * senPro[0];//�ѵ�ǰ��Ҷ�ӽڵ�����������ز�

				// Learn weights hidden -> output,���µ�ǰ��Ҷ�ӽڵ������
				for (j = 0; j < dimension; j++) 
					mapVec[j + startPos] += gradient * hiddenVec[j] * senPro[0];
			}
		}

		if(negative > 0)//����neg��ϵ
		{
			for (i = 0; i < negative + 1; i++) 
			{
				if (i == 0) 
				{
					target = position;//���Ĵ�
					label = 1;//������
				}
				else 
				{
					next_random = next_random * (unsigned long long)25214903917 + 11;
					target = table[(next_random >> 16) % table_size];

					if (target == 0)//�����˵�ǰ���Ĵ� 
						target = next_random % (vocab_size - 1) + 1;
					if (target == position) 
						continue;

					label = 0;//������
				}

				startPos = target * dimension;//��������Ϣ��mapVecNeg�еĿ�ʼλ��

				for (j = 0; j < dimension; j++) 
					innerProd += hiddenVec[j] * mapVecNeg[j + startPos];//�����ڻ������㲻��

				if (innerProd > MAX_EXP)
					gradient = (label - 1) * alpha;
				else if (innerProd < -MAX_EXP) 
					gradient = (label - 0) * alpha;
				else 
					gradient = (label - expTable[(int)((innerProd + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;//������ݶȣ�ʵ�������ݶȺ�ѧϰ���ʵĳ˻�

				for (j = 0; j < dimension; j++) 
					hiddenVecE[j] += gradient * mapVecNeg[j + startPos] * senPro[0];//��������
				for (j = 0; j < dimension; j++) 
					mapVecNeg[j + startPos] += gradient * hiddenVec[j] * senPro[0];//���¸���������
			}
		}

		// hidden -> in
		for (i = 1; i < length; i++)//��input layer��word vector���и���
		{
			cwPos = sen[i];//��ǰ�����Ĵ��ڴʱ��е�λ���±�

			for (j = 0; j < dimension; j++) 
				vector[j + cwPos * dimension] += hiddenVecE[j];
		}
	}
}

//��ͳ��Skip-garmģ�͵�ѱ��
void trainTraditionalSkipgram(long long *sen, long long length, real *hiddenVec, real *hiddenVecE, unsigned long long next_random)
{
	long long i,j,k;

	long long cwPos;//��ǰ�����Ĵ��ڴʱ��е�λ���±�
	long long startPos_1;//��ǰ�������������Ĵʵ�������ʾ��vector�еĿ�ʼλ��
	long long startPos_2;//��ǰ�������ķ�Ҷ�ӽڵ��Ȩ����mapVec�еĿ�ʼλ��

	long long target, label;

	real innerProd=0;//�ڻ�
	real gradient=0;//�ݶ�

	for (i = 1; i < length; i++) 
	{
		cwPos=sen[i];//��ǰ�����Ĵ��ڴʱ��е�λ����Ϣ

		startPos_1 = cwPos * dimension;//��ǰ�������������Ĵʵ�������ʾ�Ŀ�ʼλ��

		if (hs)//���softmax��ϵ 
		{
			for (j = 0; j < vocabulary[sen[0]].codelen; j++) //��ڵ������ǰ���Ĵʴ洢��huffman���е�·��
			{
				innerProd = 0;//�ڻ�
				startPos_2 = vocabulary[sen[0]].point[j] * dimension;//��ǰ���Ĵʵı������ķ�Ҷ�ӽڵ��Ȩ�صĿ�ʼλ��

				// Propagate hidden -> output
				for (k = 0; k < dimension; k++) 
					innerProd += vector[k + startPos_1] * mapVec[k + startPos_2];//�����������������ڻ�(û������) 

				if (innerProd <= -MAX_EXP) 
					continue;
				else if (innerProd >= MAX_EXP) 
					continue;
				else 
					innerProd = expTable[(int)((innerProd + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];

				gradient = (1 - vocabulary[sen[0]].code[j] - innerProd) * alpha;// 'g' is the gradient multiplied by the learning rate

				// Propagate errors output -> hidden
				for (k = 0; k < dimension; k++) 
					hiddenVecE[k] += gradient * mapVec[k + startPos_2];//���ز�����

				// Learn weights hidden -> output
				for (k = 0; k < dimension; k++) 
					mapVec[k + startPos_2] += gradient * vector[k + startPos_1];//���·�Ҷ�ӽڵ�Ȩ�ر�ʾ
			}
		}

		if (negative > 0)//neg��ϵ
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
			vector[k + startPos_1] += hiddenVecE[k];//������Χ�������������
	}
}

//���ڴ�ͼ��Skip-garmģ�͵�ѵ��
void trainWordLatticeSkipgram(long long *sen, real *senPro, long long length, real *hiddenVec, real *hiddenVecE, unsigned long long next_random)
{
	long long i,j,k;

	long long cwPos;//��ǰ�����Ĵ��ڴʱ��е�λ���±�
	long long startPos_1;//��ǰ�������������Ĵʵ�������ʾ��vector�еĿ�ʼλ��
	long long startPos_2;//��ǰ�������ķ�Ҷ�ӽڵ��Ȩ����mapVec�еĿ�ʼλ��

	long long target, label;

	real innerProd=0;//�ڻ�
	real gradient=0;//�ݶ�
	real sumPro=0;//�����Ĵʵĸ���ֵ��

	for(k=1;k<length;k++)//���������Ĵʵĸ���ֵ�ͣ�������һ��
		sumPro+=senPro[k];

	for (i = 1; i < length; i++) 
	{
		cwPos=sen[i];//��ǰ�����Ĵ��ڴʱ��е�λ����Ϣ

		startPos_1 = cwPos * dimension;//��ǰ�������������Ĵʵ�������ʾ�Ŀ�ʼλ��

		if (hs)//���softmax��ϵ 
		{
			for (j = 0; j < vocabulary[sen[0]].codelen; j++) //��ڵ������ǰ���Ĵʴ洢��huffman���е�·��
			{
				innerProd = 0;//�ڻ�
				startPos_2 = vocabulary[sen[0]].point[j] * dimension;//��ǰ���Ĵʵı������ķ�Ҷ�ӽڵ��Ȩ�صĿ�ʼλ��

				// Propagate hidden -> output
				for (k = 0; k < dimension; k++) 
					innerProd += vector[k + startPos_1] * mapVec[k + startPos_2];//�����������������ڻ�(û������) 

				if (innerProd <= -MAX_EXP) 
					continue;
				else if (innerProd >= MAX_EXP) 
					continue;
				else 
					innerProd = expTable[(int)((innerProd + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];

				gradient = (1 - vocabulary[sen[0]].code[j] - innerProd) * alpha;// 'g' is the gradient multiplied by the learning rate

				// Propagate errors output -> hidden
				for (k = 0; k < dimension; k++) 
					hiddenVecE[k] += gradient * mapVec[k + startPos_2] * (senPro[0]*senPro[i]/sumPro);//���ز�����

				// Learn weights hidden -> output
				for (k = 0; k < dimension; k++) 
					mapVec[k + startPos_2] += gradient * vector[k + startPos_1] * (senPro[0]*senPro[i]/sumPro);//���·�Ҷ�ӽڵ�Ȩ�ر�ʾ
			}
		}

		if (negative > 0)//neg��ϵ
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
			vector[k + startPos_1] += hiddenVecE[k];//������Χ�������������
	}
}

//ģ��ѵ���̣߳���ִ���߳�֮ǰҪ���ݴ�Ƶ����Ĵʱ��ÿ�������huffman����
void *trainModelThread(void *id) 
{
	long long i=0;
	char ch;

	char content[MAX_STRING];//���湹�����ʱ���ļ��ж���������
	bool begin=true;//��ʾ��Ҫ�����������ľ���
	int flag=1;//�жϵ�ǰ��õ���Ϣ�Ǵ���Ϣ����Ƶ����Ϣ

	long long word;//�������ӵ����ã�������ɺ��ʾ�����еĵ�ǰ����(�����Ĵ�)�ڴʱ��е�λ��
	long long sentence_length=0;//��ǰ���ӵĳ��� 
	long long pro_length=0;//��ǰ�����ж�Ӧ�ʵĸ���ֵ�����±�
	
	long long word_count = 0;//��ѵ�����ܸ���
	long long last_word_count = 0;//����ֵ���Ա���ѵ�����ϳ��ȳ���ĳ��ֵʱ�����Ϣ
	long long sen[MAX_SENTENCE_LENGTH + 1];//��ǰ����
	real senPro[MAX_SENTENCE_LENGTH + 1];//��ǰ������ÿ���ʶ�Ӧ�ĸ���ֵ
	
	long long local_iter = iter;
	unsigned long long next_random = (long long)id;
	clock_t now;
	
	real *hiddenVec = (real *)calloc(dimension, sizeof(real));//������
	real *hiddenVecE = (real *)calloc(dimension, sizeof(real));//����ۼ���
	FILE *fi = fopen(trainFile, "rb");
	
	 //ÿ���̶߳�Ӧһ���ı��������߳�id�ҵ��Լ�������ı��ĳ�ʼλ�� 
	fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
	do ch = fgetc(fin);
	while (ch != '\n')//���뵱ǰ�ַ�
	
	while (1) 
	{
		if (word_count - last_word_count > 10000) //ѵ���ʸ�������ĳ��ֵ(10000)
		{
			word_count_actual += word_count - last_word_count;//��ѵ���ʵĸ���
			last_word_count = word_count;
			
			if ((debug_mode > 1)) 
			{
				now=clock();
				printf("%cAlpha: %f Progress: %.2f%% Words/thread/sec: %.2fk ", 13, alpha,
					word_count_actual / (real)(iter * train_words + 1) * 100,
					word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
				fflush(stdout);
			}
			
			alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));//��ν���ѧϰ����
			
			if (alpha < starting_alpha * 0.0001) //��ѧϰ���ʹ�Сʱ��Ҫ���о���
				alpha = starting_alpha * 0.0001;
		}
		
		// ����һ������
		if (sentence_length == 0) 
		{
			flag = 1;
			while (1) 
			{
				readWordForSen(content, fi);//���ļ��ж�ȡ���������쵱ǰ����

				if (feof(fi)) 
					break;

				if(flag==1)//��ǰ���ļ��ж����������Ǵ�
				{
					word = searchVocab(content);
					if (word == -1)
						continue;

					word_count++;
					if (word == 0)//�����˾��ӽ�����ʶ
						break;

					//�²��������л����������Ƶ�ʵĴʣ���Ҫ���������һ�� ,��ǰ���Խ����²���
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
					senPro[pro_length] = atof(content); //���浱ǰ�ʵĸ���ֵ
					//memcpy(&senPro[pro_length], content, 20);
					pro_length++;
					assert(pro_length == sentence_length); //debug
				}	
				flag = !flag;
			}
		}//���Ӷ������
		
		if (feof(fi) || (word_count > train_words / num_threads)) //��ǰ�߳�Ӧ����Ĳ��ֽ���
		{
			word_count_actual += word_count - last_word_count;
			local_iter--;//���������ݼ�
			if (local_iter == 0) //����iter�ξͽ���
				break;
			
			word_count = 0;
			last_word_count = 0;
			sentence_length = 0;
			fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
			do ch = fgetc(fin);
			while (ch != '\n')//���뵱ǰ�ַ�
			
			continue;//��ʼ�µ�һ�ֵ���
		}
		
		word = sen[0];//���Ĵ��ڴʱ��е�λ�� 
		if (word == -1) 
			continue;
		
		for (i = 0; i < dimension; i++) //��ʼ�����㵥Ԫ
			hiddenVec[i] = 0;
		for (i = 0; i < dimension; i++) //��ʼ������ۼ���
			hiddenVecE[i] = 0;
		
		//ѵ��CBOWģ��
		trainTraditionalCBOW(sen,sentence_length,hiddenVec,hiddenVecE,next_random,word);//��ͳCBOWģ�͵�ѵ��(hs+neg)
		trainWordLatticeCBOW(sen,senPro,sentence_length,hiddenVec,hiddenVecE,next_random,word);//���ڴ�ͼ��CBOWģ�͵�ѵ��(hs+neg)
		
		//ѵ��skip-gramģ��
		trainTraditionalSkipgram(sen,sentence_length,hiddenVec,hiddenVecE,next_random);//��ͳship-gramģ�͵�ѵ��(hs+neg)
		trainWordLatticeSkipgram(sen,senPro,sentence_length,hiddenVec,hiddenVecE,next_random);//���ڴ�ͼ��ship-gramģ�͵�ѵ��(hs+neg)

		//��յ�ǰ���ӳ�����Ϣ
		sentence_length=0;
	
	}//while����
	
	fclose(fi);
	free(hiddenVec);
	free(hiddenVecE);
	pthread_exit(NULL);
}

//ģ��ѵ��
void trainModel() 
{
	long i,j;//��������
	FILE *fo;
	
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	printf("Starting training using file %s\n", trainFile);
	
	starting_alpha = alpha;//����ѧϰ���ʵĳ�ʼֵ
	if (vocabReadFile[0] != 0) 
		readVocab(); //���ļ�����ʱ�
	else
		learnVocabFromTrainFile();//��ѵ���ļ�ѧϰ�ʻ�  
	
	if (vocabSaveFile[0] != 0) 
		saveVocab();//����ʱ�
	if (outputFile[0] == 0) 
		return;
	
	initNet();//�����ʼ��
	if (negative > 0) 
		initUnigramTable();//�����������㷨�е�Ȩֵ��
	
	start = clock();//��ȡѵ���Ŀ�ʼʱ��
	for (i = 0; i < num_threads; i++) //�̴߳���
		pthread_create(&pt[i], NULL, trainModelThread, (void *)i);
		
	//�������ķ�ʽ�ȴ�ָ�����߳̽���������������ʱ�����ȴ��̵߳���Դ���ջء�����߳��Ѿ���������ô�ú������������ء�����ָ�����̱߳�����joinable�ġ�	
	for (i = 0; i < num_threads; i++) 
		pthread_join(pt[i], NULL);
	
	//��������е��̶߳��Ѿ�ִ�н�����
	
	fo = fopen(outputFile, "wb");

	// Save the word vectors
	fprintf(fo, "%lld %lld\n", vocab_size, dimension);

	for (i = 0; i < vocab_size; i++) //�����ʱ�
	{
		fprintf(fo, "%s ", vocabulary[i].word);
		for (j = 0; j < dimension; j++) //д�뵱ǰ�ʵ�ÿһά
			fprintf(fo, "%lf ", vector[i * dimension + j]);				

		fprintf(fo, "\n");
	}
	fclose(fo);
}

//���ڱ����Ŀؼ������expTable�ļ���
void init()
{
	int i;//��������

	vocabulary = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));//Ϊ�ʱ����ռ�
	hashTable = (int *)calloc(vocab_hash_size, sizeof(int));//Ϊhash�����ռ�
	expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));//Ϊ�洢sigmoid�����������ı����ռ�

	//����Ԥ���巶Χ�ڵ�sigmoid����ֵ
	for (i = 0; i < EXP_TABLE_SIZE; i++) 
	{
		expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
		expTable[i] = expTable[i] / (expTable[i] + 1); // Precompute f(x) = x / (x + 1)
	}
}

//���ڶ�ȡ�����в���,����argcΪargument count��argvΪargument value
int argPos(char *str, int argc, char **argv) 
{
	int i;//���������в���
	for (i= 1; i < argc; i++) //���ζ�ȡ����,������ִ���ļ������ڵ�λ��
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

//������
/*��ɵ����ݰ��������ղ���������ռ䣬����sigmoid����ֵ��*/
int main(int argc, char **argv) 
{
	int i=0;//���������в���

	if (argc == 1) //main������ʱû���ṩ�����������������ʾ
	{
		printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
		printf("Options:\n");
		printf("Parameters for training:\n");
		
		printf("\t-train <file>\n");//�����ļ����ѷִʵ����� 
		printf("\t\tUse text data from <file> to train the model\n");
		
		printf("\t-output <file>\n");//����ļ������������ߴʾ���
		printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
		
		printf("\t-size <int>\n");//��������ά�ȣ�Ĭ��ֵ��100 
		printf("\t\tSet size of word vectors; default is 100\n");
		
		printf("\t-sample <float>\n");//�趨�ʳ���Ƶ�ʵ���ֵ�����ڳ����ֵĴʻᱻ����²�������Ĭ��ֵ��10^3����Ч��ΧΪ(0,10^5) 
		printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
		printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
		
		printf("\t-hs <int>\n");//�Ƿ����softmax��ϵ��Ĭ����0����������  
		printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
		
		printf("\t-negative <int>\n");//��������������Ĭ����5��ͨ��ʹ��3-10��0��ʾ��ʹ�á�
		printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
		
		printf("\t-threads <int>\n");//�������߳�������Ĭ����12  
		printf("\t\tUse <int> threads (default 12)\n");
		
		printf("\t-iter <int>\n");//��������
		printf("\t\tRun more training iterations (default 5)\n");
		
		printf("\t-min-count <int>\n");//��С��ֵ�����ڳ��ִ������ڸ�ֵ�Ĵʣ��ᱻ��������Ĭ��ֵΪ5
		printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
		
		printf("\t-alpha <float>\n");//ѧϰ���ʳ�ʼֵ������skip-gramĬ����0.025������cbow��0.05 
		printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
		
		printf("\t-debug <int>\n");//debugģʽ��Ĭ����2����ʾ��ѵ�������л����������Ϣ 
		printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
		
		printf("\t-save-vocab <file>\n");//ָ������ʱ���ļ�
		printf("\t\tThe vocabulary will be saved to <file>\n");
		
		printf("\t-read-vocab <file>\n");//ָ���ʱ�Ӹ��ļ���ȡ����������ѵ����������
		printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
		
		printf("\t-cbow <int>\n");//�Ƿ����CBOW�㷨��Ĭ����1����ʾ����CBOW�㷨��
		printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
		
		printf("\nExamples:\n");//����ʹ������ 
		printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
		return 0;
	}
	
	outputFile[0] = 0;
	vocabSaveFile[0] = 0;
	vocabReadFile[0] = 0;

	if ((i = argPos((char *)"-size", argc, argv)) > 0) dimension = atoi(argv[i + 1]);//atoi���ַ���ת���͵ĺ���
	if ((i = argPos((char *)"-train", argc, argv)) > 0) strcpy(trainFile, argv[i + 1]);
	if ((i = argPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(vocabSaveFile, argv[i + 1]);
	if ((i = argPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(vocabReadFile, argv[i + 1]);
	if ((i = argPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
	if ((i = argPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
	if (cbow) alpha = 0.05;//�޸Ĳ���cbowģʽʱ�����õ�Ĭ��ѧϰ����
	if ((i = argPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
	if ((i = argPos((char *)"-output", argc, argv)) > 0) strcpy(outputFile, argv[i + 1]);
	if ((i = argPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
	if ((i = argPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
	if ((i = argPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
	if ((i = argPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
	if ((i = argPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
	if ((i = argPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
	
	init();//��ʼ��
	trainModel();//ģ��ѵ��

	return 0;
}