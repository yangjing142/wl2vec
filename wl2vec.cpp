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

#define MAX_STRING 100 //һ���ʵ���󳤶�
#define EXP_TABLE_SIZE 1000 //exp_table�Ƕ�sigmoid�����ļ��������л��棬��Ҫ��ʱ����ɣ���ʡ����ʱ�䣬���ﶨ��ֻ�ܴ洢1000�����
#define MAX_EXP 6 //����sigmoid����ֵʱ��e��ָ�������6����С��-6
#define MAX_SENTENCE_LENGTH 1000 //���ľ��ӳ��ȣ�ָ���дʵĸ���
#define MAX_CODE_LENGTH 40 //���huffman���볤��

//��ϣ�������ÿ��Ŷ�ַ�������ͻ����������Ϊ0.7������ʵ����������hash�Ĵʸ���Ϊvocab_hash_size * 0.7=2.1��10^7
const int vocab_hash_size = 30000000; // Maximum 30 * 0.7 = 21M words in the vocabulary
typedef float real; // Precision of float numbers

struct vocab_word 
{
	long long cn; //��Ƶ
	int *point; //huffman���дӸ��ڵ㵽��ǰ������Ӧ��Ҷ�ӽڵ��·���У��������з�Ҷ�ӽڵ������
	char *word, *code, codelen; //��ǰ�ʣ����Ӧ��huffman���뼰���볤��
};

char train_file[MAX_STRING], output_file[MAX_STRING]; //ѵ���ļ��� �� ����ļ���
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING]; //����ʱ���ļ� �� ���Ӹ��ļ��ж���ʱ�

struct vocab_word *vocab;//�ʱ�

/*���е�min_reduce��ָ��ReduceVocab�����л�ɾ����ƵС�����ֵ�Ĵʣ���Ϊ��ϣ���ܹ�����װ��Ĵʻ��������޵�*/
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;

//�ʻ���hash�洢���±��Ǵʵ�hashֵ�����������Ǵ��ڴʱ��е�λ��
int *vocab_hash;//�Դʻ����й�ϣ������ָ��

/*�ֱ�Ϊ�ʱ�Ĵ�С���ʱ�ĵ�ǰ��С��input layer��ÿ��word vector��ά��(Ĭ��Ϊ100)*/
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;

/*�ֱ�Ϊ��ѵ���Ĵ�����(���ڴ��Ѿ����ô�Ƶͳ�ƹ����Ĵʱ��ļ��ж�ȡ������ʱ�Ĺ����У���ֵ���Ǵ�Ƶ�ۼӣ��Դ�ͳ�Ƹ�����
���ڴӵ�����ѵ�������ж�ȡ�����Լ�����ʱ�ʱ������Ҫÿ��һ���ʾͽ���++������ͳ�ƴʸ���)
��ѵ���Ĵʸ���������������ѵ���ļ��Ĵ�С(��ftell��������)������ʱ��������*/
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;

/*����starting_alpha���ڱ���ѧϰ���ʵĳ�ʼֵ����Ϊalpha����ģ��ѵ���ķ��򴫲��������Զ�����*/
real alpha = 0.025, starting_alpha, sample = 1e-3;//ѧϰ���ʳ�ʼ��Ϊ0.025�����Ƕ���skip-gram���Ե�

/*�ֱ�Ϊinput layer�Ĵ��������У�softmax��hidden layer��huffman tree�з�Ҷ�ӽڵ��ӳ��Ȩ�أ�negative sampling��hidden layer�����������ӳ���Ȩ�أ�
����sigmoid����ֵ�ı�*/
real *syn0, *syn1, *syn1neg, *expTable;

clock_t start;//����ʼʱ��

int hs = 0, negative = 5;//ǰ�ߣ��Ƿ����softmax��ϵ�ı�ʶ�����ߣ�������������
const int table_size = 1e8;//�ڸ������������Ⱦ����ʷ�
int *table;//������

//�����������㷨�е�Ȩֵ�ֲ���
void InitUnigramTable() 
{
	int a, i;//ѭ������

	double train_words_pow = 0;//�ʵ�Ȩ����ֵ����������һ������
	double d1, power = 0.75;//��ÿ���ʶ���Ȩֵ��ʱ����ֱ��ȡcount�����Ǽ����˦����ݣ��˴���ȡ��0.75

	table = (int *)malloc(table_size * sizeof(int));//Ϊtable����ռ�

	for (a = 0; a < vocab_size; a++) //�����ʻ������train_words_pow
		train_words_pow += pow(vocab[a].cn, power);

	i = 0;//�����ʱ�
	d1 = pow(vocab[i].cn, power) / train_words_pow;//��ʼ���ѱ����Ĵʵ�ȨֵռȨ����ֵ�ı�������ʼֵΪ��һ���ʵ�Ȩֵ�������е�ռ��

	//����table�����Ⱦ��ʷ�ͶӰ���Դʱ��дʵ�Ȩ��ֵΪ��׼���еķǵȾ��ʷ��У���ʱtable��ÿ����Ԫ�б����������ͶӰ���ķǵȾ��ʷֵĶ���
	for (a = 0; a < table_size; a++)
	{
		table[a] = i;//��ǰtable�еĵȾ����ʷֶ�a��Ӧ���ǷǵȾ��ʷֶ�i

		if (a / (double)table_size > d1) 
		{
			i++;//����ʱ��е���һ��λ��
			d1 += pow(vocab[i].cn, power) / train_words_pow;//����ǰ�ʵ�ȨֵռȨ����ֵ�ı������뵽���ۼƴʵĵ�ǰ��ֵ��
		}

		if (i >= vocab_size) //�ж��Ƿ񳬹��ʻ��Ĵ�С
			i = vocab_size - 1;
	}
}

//���ļ��ж�ȡ�����Ĵʣ�����ո�+tab+EOL(end of line)�ǵ��ʱ߽�
//word���ڱ����ȡ���ĵ���
void ReadWord(char *word, FILE *fin) 
{
	int a = 0, ch;//a���ڼ���word���ַ���λ�ã�ch���ڱ��浱ǰ��ȡ�����ַ�

	while (!feof(fin))//�ж��Ƿ񵽴��ļ���β 
	{
		ch = fgetc(fin);//���뵥��

		if (ch == 13) //��ӦEOF����Ҫ������һ��
			continue;

		if ((ch == ' ') || (ch == '\t') || (ch == '\n'))//�������ʱ߽� 
		{
			if (a > 0) //word���Ѿ�����������
			{
				if (ch == '\n') 
					ungetc(ch, fin);//��һ���ַ��˻ص��������У�����˻ص��ַ�������һ����ȡ�ļ����ĺ���ȡ��
				break;
			}

			if (ch == '\n') 
			{
				strcpy(word, (char *)"</s>");//���sentence�����ı��
				return;
			} 
			else 
				continue;
		}

		word[a] = ch;
		a++;

		if (a >= MAX_STRING - 1)//�ضϹ����ĵ���
			a--;
	}

	word[a] = 0;//���Ͻ�������\0��
}

//����һ���ʵ�hashֵ��һ���ʶ�Ӧһ��hashֵ����һ��hashֵ���Զ�Ӧ����ʣ�����ϣ��ͻ
int GetWordHash(char *word) 
{
	unsigned long long a, hash = 0;
	
	for (a = 0; a < strlen(word); a++) //����word�е�ÿ���ַ�������257���ƹ��쵱ǰword��key value
		hash = hash * 257 + word[a];
	
	hash = hash % vocab_hash_size;//����ɢ�к���ΪH(key)=key % m��������������
	
	return hash;
}

//���ص�ǰ���ڴʱ��е�λ�ã���������ھͷ���-1
int SearchVocab(char *word) 
{
	unsigned int hash = GetWordHash(word);//��õ�ǰ�ʵ�hashֵ
	
	while (1) 
	{
		if (vocab_hash[hash] == -1)//��ǰhashֵ��hash���в����ڣ�����ǰ�ʲ����ڣ�����-1
			return -1;
			
		//�õ�ǰ����hashֵ�ڴʱ����ҵ��Ĵ��뵱ǰ�ʶԱȣ������ͬ�����ʾ��ǰ��hashֵ��Ӧ��ȷ
		if (!strcmp(word, vocab[vocab_hash[hash]].word)) 
			return vocab_hash[hash];
		
		hash = (hash + 1) % vocab_hash_size;//���Ŷ�ַ��
	}
	return -1;
}

// ���ļ����ж�ȡһ���ʣ�������������ڴʻ���е�λ��
int ReadWordIndex(FILE *fin) 
{
	char word[MAX_STRING];
	
	ReadWord(word, fin);//���ļ��ж�ȡ����
	
	if (feof(fin)) 
		return -1;
	
	return SearchVocab(word);
}

//��һ������ӵ��ʱ���  
int AddWordToVocab(char *word) 
{
	unsigned int hash, length = strlen(word) + 1;//��ǰ�ʵ�hashֵ��length
	
	if (length > MAX_STRING) 
		length = MAX_STRING;
	
	vocab[vocab_size].word = (char *)calloc(length, sizeof(char));//Ϊ������Ĵʵ�λ�÷���ռ�
	
	strcpy(vocab[vocab_size].word, word);
	vocab[vocab_size].cn = 0;//�����Ƶ��Ϊ0
	vocab_size++;
	
	// Reallocate memory if needed
	/*�����ǰ�ʱ��ʵ�ʴ�С�ӽ�������ƴ�С�Ļ����������ݣ�ÿ�οմ�1000
	���ڵ�ǰ���·�����ڴ�ռ��ԭ���Ĵ�����ԭ���ݱ����������ӵĿռ䲻��ʼ��*/
	if (vocab_size + 2 >= vocab_max_size) 
	{
		vocab_max_size += 1000;
		vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
	}
	
	hash = GetWordHash(word);//��ȡ��ǰ�¼���Ĵʵ�hashֵ
	while (vocab_hash[hash] != -1) //����ǰλ���Ѿ�����������ʣ������˳�ͻ
		hash = (hash + 1) % vocab_hash_size;//ʹ�ÿ��ŵ�ַ�������ͻ
	
	//�ɴʵ�hashֵ�ҵ������ڴʱ������λ�á����ڹ�ϣ���е�ǰ�ʵ�hashֵ��ָ��ĵ�Ԫ�д�ŵ��Ǹô��ڴʱ��е�λ���±�
	vocab_hash[hash] = vocab_size - 1;
	
	return vocab_size - 1;//�����¼���Ĵ��ڴʱ��е��±�
}

//�Ƚ������ʴ�Ƶ�Ĵ�С
int VocabCompare(const void *a, const void *b) 
{
	return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

//���ʱ��մ�Ƶ��������
void SortVocab() 
{
	int a, size;
	unsigned int hash;
	
	// Sort the vocabulary and keep </s> at the first position
	/*qsort������&vocab[1]Ϊ������������ͷָ��(���ﲻ��&vocab[0]��Ϊ�˱���</s>һֱ����λ)��
	vocab_size - 1Ϊ����������Ĵ�С��sizeof(struct vocab_word)Ϊ����Ԫ�ش�С��VocabCompareΪ�жϴ�С�ĺ�����ָ��*/
	qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
	
	for (a = 0; a < vocab_hash_size; a++) //���³�ʼ��hash��
		vocab_hash[a] = -1;
	
	size = vocab_size;//������ǰ�ʱ��ԭʼ��С
	train_words = 0;//��ѵ���ĴʵĴ�Ƶ�Ӻ�ֵ��ʱΪ0
	
	for (a = 0; a < size; a++) //������ǰ�ʱ����е����д�
	{
		//�жϵ�ǰ�ʵĴ�Ƶ�Ƿ�С����Сֵ���ǵĻ��ʹӴʱ���ȥ��
		if ((vocab[a].cn < min_count) && (a != 0)) 
		{
			vocab_size--;
			free(vocab[a].word);
		} 
		else 
		{
			// �������Ҫ���¼���hash����
			hash=GetWordHash(vocab[a].word);
			while (vocab_hash[hash] != -1) //������ͻ
				hash = (hash + 1) % vocab_hash_size;//���ŵ�ַ��
			vocab_hash[hash] = a;//hash�б��浱ǰ���ڴʱ��е�λ��
			
			train_words += vocab[a].cn;
		}
	}
	
	vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));//׷�ӿռ�
	
	//Ϊ�������ṹ����ռ�
	for (a = 0; a < vocab_size; a++) 
	{
		vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
		vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
	}
}

//�Ƴ���Ƶ��С�Ĵʣ������ʱ�
void ReduceVocab() 
{
	int a, b = 0;
	unsigned int hash;

	for (a = 0; a < vocab_size; a++) //�����ʱ�
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
	
	vocab_size = b;//��ǰ�ʱ��������������ʵ�ʴ�С
	for (a = 0; a < vocab_hash_size; a++)
		vocab_hash[a] = -1;
	for (a = 0; a < vocab_size; a++)//���¹�������������Ĵʱ��hash��
	{
		hash = GetWordHash(vocab[a].word);
		while (vocab_hash[hash] != -1)
			hash = (hash + 1) % vocab_hash_size;
		vocab_hash[hash] = a;
	}
	
	fflush(stdout);
	min_reduce++;//ע��������Ƶ����ֵ
}

//ͨ����Ƶ��Ϣ����huffman������Ƶ�ߵĴʵ�huffman��������
void CreateBinaryTree() 
{
	/*pos1��pos2���ڱ����ڹ���huffman���Ĺ�������ѡ��Ȩ����С�Ľڵ��ڴʱ��е��±�
	min1i,min2i���ڱ��湹���������ҵ�����СȨ��*/
	long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
	char code[MAX_CODE_LENGTH];
	
	//�����ڹ���huffman���������õ��Ľڵ��Ƶ��Ϣ����Ϊ��Ҷ�ӽڵ���n�����򹹽��������з�Ҷ�ӽڵ������n-1������2*n+1���ڵ�
	long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));//��������֧�ı�����Ϣ����������С�ߵı���Ϊ0����С�ߵı���Ϊ1
	long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));//���游�׽ڵ��ڴʱ��е�λ����Ϣ
	
	for (a = 0; a < vocab_size; a++) //��ʼ����ǰҶ�ӽڵ��Ƶ��Ϣ(��Ȩ��)
		count[a] = vocab[a].cn;
	for (a = vocab_size; a < vocab_size * 2; a++) //����δ֪�ķ�Ҷ�ӽڵ��Ȩ�أ�ȫ����һ��������ʼ��
		count[a] = 1e15;
	
	pos1 = vocab_size - 1;
	pos2 = vocab_size;
	
	//����huffman��
	for (a = 0; a < vocab_size - 1; a++)//�����ʱ� 
	{
		// �ҳ�ĿǰȨֵ��С�������ڵ� 
		if (pos1 >= 0) //��һ��Ȩֵ��С�Ľڵ�  
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
		
		if (pos1 >= 0)//�ڶ���Ȩֵ��С�Ľڵ�  
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
		
		count[vocab_size + a] = count[min1i] + count[min2i];//���¹���ķ�Ҷ�ӽڵ��Ȩ�ؼ��뵽Ȩ�ؼ���
		
		//Ϊ���ӽڵ㱣�浱ǰ���׽ڵ��ڴʱ��е�λ��
		parent_node[min1i] = vocab_size + a;
		parent_node[min2i] = vocab_size + a;
		
		binary[min2i] = 1;//�ڵ����Ϊ1��֮ǰĬ����0��  
	}
	
	//������õ�huffman�������ʱ��е�ÿ����
	for (a = 0; a < vocab_size; a++) 
	{
		b = a;//���ڱ������Ĵ��ڴʱ��е�λ���±�
		i = 0;//��¼���ݵķ�֧����
		
		while (1) 
		{
			code[i] = binary[b];
			point[i] = b;
			i++;
			
			//@todo �ж�i�Ƿ����code_len,���ھͼ�¼�����˳�
			
			b = parent_node[b];//��������ڵ����
			if (b == vocab_size * 2 - 2) //�Ѿ�������ڵ�
				break; 
		}
		
		vocab[a].codelen = i;
		vocab[a].point[0] = vocab_size - 2;//�����ڵ����·����
		for (b = 0; b < i; b++) 
		{
			vocab[a].code[i - b - 1] = code[b];
			vocab[a].point[i - b] = point[b] - vocab_size;
		}
	}
	
	//�ͷſռ�
	free(count);
	free(binary);
	free(parent_node);
}

//��ѵ���ļ��ж�ȡ����������ʱ�
//���ļ���û��ͳ��ÿ�����ʵĴ�Ƶ
//ͬʱ���ѵ�������ļ��Ĵ�С
void LearnVocabFromTrainFile()
{
	char word[MAX_STRING];//��ѵ���ļ��ж����Ĵ�
	FILE *fin;
	long long a, i;
	
	for (a = 0; a < vocab_hash_size; a++) //hash���ʼ��
		vocab_hash[a] = -1;
	
	fin = fopen(train_file, "rb");//���ļ�
	if (fin == NULL) 
	{
		printf("ERROR: training data file not found!\n");
		exit(1);
	}
	
	vocab_size = 0;//�ʱ�ǰʵ�ʴ�СΪ0
	AddWordToVocab((char *)"</s>");//��֤</s>����ǰ��
	
	while (1) 
	{
		ReadWord(word, fin);//��ѵ���ļ��ж�ȡһ����
		if (feof(fin)) 
			break;
		train_words++;//��ѵ�����ʸ�����1
		
		if ((debug_mode > 1) && (train_words % 100000 == 0)) 
		{
			printf("%lldK%c", train_words / 1000, 13);
			fflush(stdout);
		}
		
		i = SearchVocab(word);//���ظô��ڴʱ��е�λ��
		if (i == -1)//��ʱ�ôʲ������ڴʱ�֮�У��ͽ������ʱ�֮��  
		{
			a = AddWordToVocab(word);
			vocab[a].cn = 1;
		} 
		else 
			vocab[i].cn++;//���´�Ƶ
		
		if (vocab_size > vocab_hash_size * 0.7) //����ʱ�̫�Ӵ󣬾������ʱ�
			ReduceVocab();
	}
	
	SortVocab();//���ݴ�Ƶ���ʱ�����
	
	if (debug_mode > 0) 
	{
		printf("Vocab size: %lld\n", vocab_size);
		printf("Words in train file: %lld\n", train_words);
	}
	
	file_size = ftell(fin);//���ѵ���ļ��Ĵ�С
	fclose(fin);
}

//����ʱ�
void SaveVocab() 
{
	long long i;
	FILE *fo = fopen(save_vocab_file, "wb");
	for (i = 0; i < vocab_size; i++) 
		fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
	fclose(fo);
}

//���ļ��ж�ȡ�ʱ����ļ��Ѿ�ͳ�ƺ���ÿ���ʵĴ�Ƶ
//ͬʱ���ѵ�������ļ��Ĵ�С
void ReadVocab() 
{
	long long a, i = 0;
	char c;
	char word[MAX_STRING];//���ڱ��浱ǰ���ļ��ж����Ĵ�
	
	FILE *fin = fopen(read_vocab_file, "rb");//�򿪴ʱ��ļ�
	if (fin == NULL) 
	{
		printf("Vocabulary file not found\n");
		exit(1);
	}
	
	//hash���ʼ��
	for (a = 0; a < vocab_hash_size; a++) 
		vocab_hash[a] = -1;
	
	vocab_size = 0;//��ǰ�ʱ��ʵ�ʴ�СΪ0
	while (1) 
	{
		ReadWord(word, fin);//���ļ��ж���һ����
		if (feof(fin)) 
			break;
		
		a = AddWordToVocab(word);//����ǰ���ļ��ж�ȡ���Ĵʼ��뵽�ʱ��У��������ڴʱ��е�λ��
		
		/*��stream����������ȡ�ܹ�ƥ��format��ʽ���ַ��������б��ж�Ӧ�ı�����
		finΪ�ļ�ָ�룬"%lld%c"��format���ʽ,Ҫ�ɡ��ո񡱡����ǿո񡱼���ת��������ɣ���ʽΪ%[*][width][modifiers]type
		��ʾ���ļ��ж�ȡlld��С�ġ���ת����%c���ַ���Ҳ����char��int�������䱣�浽����ĵ�ַ��
		���������롰format���С�ת��������Ӧ������ַ���б�����ַ���ö��Ÿ�����*/
		fscanf(fin, "%lld%c", &vocab[a].cn, &c);
		
		i++;
	}
	
	SortVocab();//���ݴ�Ƶ����
	
	if (debug_mode > 0) 
	{
		printf("Vocab size: %lld\n", vocab_size);
		printf("Words in train file: %lld\n", train_words);
	}
	
	fin = fopen(train_file, "rb");//��ȡѵ������
	if (fin == NULL) 
	{
		printf("ERROR: training data file not found!\n");
		exit(1);
	}
	
	fseek(fin, 0, SEEK_END);
	file_size = ftell(fin);//��õ�ǰѵ�������ļ��Ĵ�С
	fclose(fin);
}

//����ģ�ͳ�ʼ��
//Ϊ���Ƿ���ռ䣬��ʼ������������ͬʱ����huffman��
void InitNet() 
{
	long long a, b;
	unsigned long long next_random = 1;
	
	//posix_memalign() �ɹ�ʱ�᷵��size�ֽڵĶ�̬�ڴ棬��������ڴ�ĵ�ַ��alignment(������128)�ı���,��ʱ��������ڴ��СΪ�ʱ��С * ����ÿ���ʵ�ά�� * ʵ����С
	a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));//Ԥ�����ڴ�ķ���
	if (syn0 == NULL) //��̬�ڴ����ʧ��
	{
		printf("Memory allocation failed\n"); 
		exit(1);
	}
	
	if (hs) //����softmax��ϵ
	{
		a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));//Ϊhuffman tree�з�Ҷ�ӽڵ��Ȩ�ط��䶯̬�ڴ�
		if (syn1 == NULL) 
		{
			printf("Memory allocation failed\n"); 
			exit(1);
		}
		
		for (a = 0; a < vocab_size; a++) 
		{
			for (b = 0; b < layer1_size; b++)
				syn1[a * layer1_size + b] = 0;//ÿ����Ҷ�ӽڵ��Ȩ����Ϣ��syn1�����о���1��
		}
	}
	
	if (negative>0) //���и�����������
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
	
	for (a = 0; a < vocab_size; a++) //�����ʼ��input layer�е�ÿ��word vector ������syn0�о���1��
	{
		for (b = 0; b < layer1_size; b++) 
		{
		next_random = next_random * (unsigned long long)25214903917 + 11;
		syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
		}
	}
	
	CreateBinaryTree();//����huffman�����Դʱ��е�ÿ���ʽ��б���
}

//ģ��ѵ���̣߳���ִ���߳�֮ǰҪ���ݴ�Ƶ����Ĵʱ��ÿ�������huffman����
void *TrainModelThread(void *id) 
{
	/*cwΪ��ǰ���Ĵʵ������Ĵʸ���
	word �������ӵ����ã�������ɺ��ʾ�����еĵ�ǰ����(�����Ĵ�)�ڴʱ��е�λ��
	last_word ��һ�������ڴʱ��е�λ�ã�����ɨ�贰��
	sentence_length ��ǰ���ӵĳ���
	sentence_position ��ǰ���Ĵ��ڵ�ǰ�����е�λ��*/
	long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
	
	/*word_count ��ѵ�����ܸ���
	last_word_count ����ֵ���Ա���ѵ�����ϳ��ȳ���ĳ��ֵʱ�����Ϣ
	sen ��ǰ����*/
	long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
	
	long long l1, l2, c, target, label, local_iter = iter;
	unsigned long long next_random = (long long)id;
	real f, g;
	clock_t now;
	
	real *neu1 = (real *)calloc(layer1_size, sizeof(real));//������
	real *neu1e = (real *)calloc(layer1_size, sizeof(real));//����ۼ���
	FILE *fi = fopen(train_file, "rb");
	
	 //ÿ���̶߳�Ӧһ���ı��������߳�id�ҵ��Լ�������ı��ĳ�ʼλ�� 
	fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
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
			while (1) 
			{
				word = ReadWordIndex(fi);//���ļ����ж�ȡһ���ʣ�������������ڴʱ��е�λ��
				
				if (feof(fi)) 
					break;
				if (word == -1)
					continue;
				
				word_count++;
				if (word == 0) //������β
					break;
				
				//�²��������л����������Ƶ�ʵĴʣ���Ҫ���������һ��
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
			
			sentence_position = 0;//��ʼ��Ϊ0��ʾȡ��ǰ���еĵ�һ����Ϊ���Ĵ�
		}
		
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
			continue;//��ʼ�µ�һ�ֵ���
		}
		
		word = sen[sentence_position];//���Ĵ��ڴʱ��е�λ�� 
		if (word == -1) 
			continue;
		
		for (c = 0; c < layer1_size; c++) //��ʼ�����㵥Ԫ
			neu1[c] = 0;
		for (c = 0; c < layer1_size; c++) 
			neu1e[c] = 0;
		
		next_random = next_random * (unsigned long long)25214903917 + 11;
		b = next_random % window;
		//���ڴ�С����window-b
		
		//ѵ��CBOWģ��
		if (cbow) 
		{ 
			//��input layer�� hidden layer ��ӳ��
			cw = 0;
			for (a = b; a < window * 2 + 1 - b; a++) //ɨ�����Ĵʵ����ҵļ�����
			{
				if (a != window) //��ʱ��δɨ�赽��ǰ���Ĵ�
				{
					c = sentence_position - window + a;//��ǰɨ�赽�������Ĵ��ڴʱ��е�λ��
					if (c < 0) 
						continue;
					if (c >= sentence_length) 
						continue;
					
					last_word = sen[c];//��¼��һ�����ڴʱ��е�λ�ã����㸨������ɨ��
					if (last_word == -1) 
						continue;
					
					for (c = 0; c < layer1_size; c++) //layer1_size��������ά�ȣ�Ĭ��ֵ��100 
						neu1[c] += syn0[c + last_word * layer1_size];
					
					cw++;//�����Ĵ�����1
				}
			}
			
			//��ʱ���ز㵥Ԫ�Ѿ��������
			
			if (cw) 
			{
				for (c = 0; c < layer1_size; c++) //��ƽ��
					neu1[c] /= cw;
				
				if (hs) 
				{
					for (d = 0; d < vocab[word].codelen; d++) //��ڵ������ǰ���Ĵʴ洢��huffman���е�·��
					{
						f = 0;
						l2 = vocab[word].point[d] * layer1_size;//��ǰ���Ĵʵı������ķ�Ҷ�ӽڵ��Ȩ�صĿ�ʼλ��
						
						// Propagate hidden -> output
					    for (c = 0; c < layer1_size; c++) 
							f += neu1[c] * syn1[c + l2];//�����ڻ�
						
						if (f <= -MAX_EXP) //�ڻ����ڷ�Χ����ֱ�Ӷ��� 
							continue;
						else if (f >= MAX_EXP) 
							continue;
						else 
							f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];//�ڻ�֮��sigmoid����
						
						g = (1 - vocab[word].code[d] - f) * alpha;// 'g' is the gradient multiplied by the learning rate
						
						// Propagate errors output -> hidden��layer1_size��������ά��
						for (c = 0; c < layer1_size; c++) //���򴫲�����huffman���������ز�
							neu1e[c] += g * syn1[c + l2];//�ѵ�ǰ��Ҷ�ӽڵ�����������ز�
						
						// Learn weights hidden -> output,���µ�ǰ��Ҷ�ӽڵ������
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
							target = word;//���Ĵ�
							label = 1;//������
						}
						else 
						{
						    next_random = next_random * (unsigned long long)25214903917 + 11;
						    target = table[(next_random >> 16) % table_size];
						
						    if (target == 0) 
							    target = next_random % (vocab_size - 1) + 1;
						    if (target == word) 
							    continue;
						
						    label = 0;//������
					    }
						
						l2 = target * layer1_size;
					    f = 0;
					    for (c = 0; c < layer1_size; c++) 
						    f += neu1[c] * syn1neg[c + l2];//�ڻ������㲻��
					
					    if (f > MAX_EXP)
						    g = (label - 1) * alpha;
					    else if (f < -MAX_EXP) 
						    g = (label - 0) * alpha;
					    else 
						    g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
					
					    for (c = 0; c < layer1_size; c++) 
						    neu1e[c] += g * syn1neg[c + l2];//��������
					    for (c = 0; c < layer1_size; c++) 
						    syn1neg[c + l2] += g * neu1[c];//���¸���������
				    }
				}
				
				// hidden -> in
				for (a = b; a < window * 2 + 1 - b; a++)//��input layer��word vector���и���
				{
					if (a != window) //û��������ǰ���Ĵ�
					{
						c = sentence_position - window + a;//��ǰ�����Ĵ��ڴʵ��е�λ��
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
		else //ѵ��skip-gramģ��
		{ 
			for (a = b; a < window * 2 + 1 - b; a++)  //��
			{
				if (a != window)//ɨ�������Ĵ� //��
				{
					c = sentence_position - window + a;
					if (c < 0) 
						continue;
					if (c >= sentence_length)
						continue;
					
					last_word = sen[c];
					
					if (last_word == -1) 
						continue;
					
					l1 = last_word * layer1_size;//��ǰ�������������Ĵʵ�������ʾ�Ŀ�ʼλ��
					for (c = 0; c < layer1_size; c++) //���³�ʼ��
						neu1e[c] = 0;
					
					// HIERARCHICAL SOFTMAX
					if (hs) 
					{
						for (d = 0; d < vocab[word].codelen; d++) //��ڵ������ǰ���Ĵʴ洢��huffman���е�·��
						{
							f = 0;
							l2 = vocab[word].point[d] * layer1_size;//��ǰ���Ĵʵı������ķ�Ҷ�ӽڵ��Ȩ�صĿ�ʼλ��
						
						    // Propagate hidden -> output
							for (c = 0; c < layer1_size; c++) 
								f += syn0[c + l1] * syn1[c + l2];//�����������������ڻ�(û������) 
							
							if (f <= -MAX_EXP) 
								continue;
							else if (f >= MAX_EXP) 
								continue;
							else 
								f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
							
							g = (1 - vocab[word].code[d] - f) * alpha;// 'g' is the gradient multiplied by the learning rate
							
							// Propagate errors output -> hidden
							for (c = 0; c < layer1_size; c++) 
								neu1e[c] += g * syn1[c + l2];//���ز�����
							
							// Learn weights hidden -> output
							for (c = 0; c < layer1_size; c++) 
								syn1[c + l2] += g * syn0[c + l1];//���·�Ҷ�ӽڵ�Ȩ�ر�ʾ
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
						syn0[c + l1] += neu1e[c];//������Χ�������������
				}
			}
		}
		
		sentence_position++;//ѡ����һ�����Ĵ�
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

//ģ��ѵ��
void TrainModel() 
{
	long a, b, c, d;
	FILE *fo;
	
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	printf("Starting training using file %s\n", train_file);
	
	starting_alpha = alpha;//����ѧϰ���ʵĳ�ʼֵ
	if (read_vocab_file[0] != 0) 
		ReadVocab(); //���ļ�����ʱ�
	else
		LearnVocabFromTrainFile();//��ѵ���ļ�ѧϰ�ʻ�  
	
	if (save_vocab_file[0] != 0) 
		SaveVocab();//����ʱ�
	if (output_file[0] == 0) 
		return;
	
	InitNet();//�����ʼ��
	if (negative > 0) 
		InitUnigramTable();
	
	start = clock();//��ȡѵ���Ŀ�ʼʱ��
	for (a = 0; a < num_threads; a++) //�̴߳���
		pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
		
	//�������ķ�ʽ�ȴ�ָ�����߳̽���������������ʱ�����ȴ��̵߳���Դ���ջء�����߳��Ѿ���������ô�ú������������ء�����ָ�����̱߳�����joinable�ġ�	
	for (a = 0; a < num_threads; a++) 
		pthread_join(pt[a], NULL);
	
	//��������е��̶߳��Ѿ�ִ�н�����
	
	fo = fopen(output_file, "wb");
	if (classes == 0) //����Ҫ���ֻ࣬��Ҫ���������
	{
		// Save the word vectors
		fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
		
		for (a = 0; a < vocab_size; a++) //�����ʱ�
		{
			fprintf(fo, "%s ", vocab[a].word);
			if (binary)//��������ʽ���
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
	else //ʹ��k-means���о���
	{
		// Run K-means on the word vectors
		int clcn = classes, iter = 10, closeid;
		int *centcn = (int *)malloc(classes * sizeof(int));//����������
		int *cl = (int *)calloc(vocab_size, sizeof(int));//�ʵ�����ӳ�� 
		real closev, x;
		real *cent = (real *)calloc(classes * layer1_size, sizeof(real));//�������� 
		
		for (a = 0; a < vocab_size; a++) 
			cl[a] = a % clcn;
		for (a = 0; a < iter; a++) 
		{
			for (b = 0; b < clcn * layer1_size; b++) 
				cent[b] = 0;//��������
			for (b = 0; b < clcn; b++) 
				centcn[b] = 1;
			for (c = 0; c < vocab_size; c++) 
			{
				for (d = 0; d < layer1_size; d++) 
					cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
				centcn[cl[c]]++;//���������1
			}
			
			for (b = 0; b < clcn; b++)//����������� 
			{
				closev = 0;
				for (c = 0; c < layer1_size; c++) 
				{
					cent[layer1_size * b + c] /= centcn[b];//��ֵ�������µ����� 
					closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
				}
				
				closev = sqrt(closev);
				for (c = 0; c < layer1_size; c++) 
					cent[layer1_size * b + c] /= closev;/*�����������Ľ��й�һ��������*/  
			}
			
			for (c = 0; c < vocab_size; c++)//�����д������·��� 
			{
				closev = -10;
				closeid = 0;
				for (d = 0; d < clcn; d++) 
				{
					x = 0;
					for (b = 0; b < layer1_size; b++) 
						x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];//�ڻ�
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

//���ڶ�ȡ�����в���,����argcΪargument count��argvΪargument value
int ArgPos(char *str, int argc, char **argv) 
{
	int a;
	for (a = 1; a < argc; a++) //���ζ�ȡ����,������ִ���ļ������ڵ�λ��
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

//������
/*��ɵ����ݰ��������ղ���������ռ䣬����sigmoid����ֵ��*/
int main(int argc, char **argv) 
{
	int i;//���������в���
	
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
		
		printf("\t-window <int>\n");//���ڴ�С��Ĭ����5
		printf("\t\tSet max skip length between words; default is 5\n");
		
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
		
		printf("\t-classes <int>\n");//�������𣬶����Ǵ�������Ĭ��ֵΪ0�������������
		printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
		
		printf("\t-debug <int>\n");//debugģʽ��Ĭ����2����ʾ��ѵ�������л����������Ϣ 
		printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
		
		printf("\t-binary <int>\n");//�Ƿ���binaryģʽ�������ݣ�Ĭ����0����ʾ��һ�㲻��������ʽ��
		printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
		
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
	
	output_file[0] = 0;
	save_vocab_file[0] = 0;
	read_vocab_file[0] = 0;
	
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);//atoi���ַ���ת���͵ĺ���
	if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
	if (cbow) alpha = 0.05;//�޸Ĳ���cbowģʽʱ�����õ�Ĭ��ѧϰ����
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
	
	//Ϊ�ʱ����vocab_max_size��vocab_word���ʹ�С�Ŀռ䣬��vocab_max_size*sizeof(vocab_word)���ÿռ�ĳ�ʼ����Ϊ0�ֽ�
	vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
	//��Ϊhash�����Ϊ��Ԫ�ؼ���ܴ��ڿ�϶�����Ա����Է���ռ��СΪhash��(vocab_hash_size) * int�Ĵ�С
	vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
	expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));//Ϊ�洢sigmoid�����������ı����ռ�
	
	/*����sigmoid����ֵ����MAX_EXP����Ϊ6��������СֵΪָ��Ϊ-6ʱ����exp^-6/(1+exp^-6)=1/(1+exp^6);
	���ֵΪָ��Ϊ6ʱ����exp^6/(1+exp^6)=1/(1+exp^-6)*/
	for (i = 0; i < EXP_TABLE_SIZE; i++) 
	{
		//expTable[i] = exp((i/ 1000 * 2-1) * 6) �� e^-6 ~ e^6 
		expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
		//expTable[i] = 1/(1+e^6) ~ 1/(1+e^-6)�� 0.01 ~ 1 ������
		expTable[i] = expTable[i] / (expTable[i] + 1); // Precompute f(x) = x / (x + 1)
	}
	
	TrainModel();
	return 0;
}