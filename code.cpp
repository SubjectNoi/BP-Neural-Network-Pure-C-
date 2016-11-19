#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include <ctime>
#include <cstdlib>
using namespace std;
#define LAYER        3
#define NODE         10
#define A            30.0
#define B            10.0   //A and B are parameter of sigmoid function
#define ITER         100000  //Max training rounds
#define ETA_W        0.0035 //adjust rate of weight
#define ETA_B        0.001  //adjust rate of threshold
#define ERROR        0.0005  //error of a sample
#define ACCU         0.0005  //error of a round of iter
#define Type double
struct Data{
	vector <Type> x;        //input data
	vector <Type> y;        //output data
};

class BP {
public:
	void GetData(const vector<Data>);
	void Train();
	vector <Type> ForeCast(const vector<Type>);
private:
	void InitNetwork();       //initialization
	void GetNums();           //get number of nodes of input, hidden, output layer
	void Forward();           //input -> output
	void Reverse(int);        //output -> input
	void Update();            //update new weight and threshold
	void CalcDelta(int);      //caculate adjust of W and B
	Type GetError(int);       //caculate error of a sample
	Type GetAccu();           //caculate all samples
	Type Sigmoid(const Type); //caculate sigmoid
private:
	int in_num;               //number of Node in input layer
	int out_num;              //number of Node in output layer
	int hidden_num;           //number of Node in hidden layer
	vector <Data> data;       //input and output data
	Type Weight[LAYER][NODE][NODE];
	Type Threshold[LAYER][NODE];

	Type X[LAYER][NODE];      //output after sigmoid
	Type Delta[LAYER][NODE];  //delta of delta learning rules
};

void BP::GetData(const vector<Data> __data) {
	data = __data;
}

void BP::Train() {
	GetNums();
	InitNetwork();
	int num = data.size();
	for (int iter = 0; iter <= ITER; iter++) {
		for (int cnt = 0; cnt < num; cnt++) {
			for (int i = 0; i < in_num; i++) {
				X[0][i] = data.at(cnt).x[i];
			}
			while (true) {
				Forward();
				if (GetError(cnt) < ERROR) break;
				Reverse(cnt);
			}
		}
		Type accu = GetAccu();
		printf("All samples' Accuracy: %lf\n", accu);
		if (accu < ACCU) break;
	}
}

vector<Type> BP::ForeCast(const vector<Type> data) {
	int n = data.size();
	assert(n == in_num);
	for (int i = 0; i < in_num; i++) {
		X[0][i] = data[i];
	}
	Forward();
	vector <Type> vis;
	for (int i = 0; i < out_num; i++) {
		vis.push_back(X[2][i]);
	}
	return vis;
}

void BP::GetNums() {
	in_num = data[0].x.size();
	out_num = data[0].y.size();
	hidden_num = (int)sqrt((in_num + out_num) * 1.0) + 5;
	if (hidden_num > NODE) hidden_num = NODE;
}

void BP::InitNetwork() {
	srand(time(0));
	for (int i = 0; i < LAYER; i++) {
		for (int j = 0; j < NODE; j++) {
			for (int k = 0; k < NODE; k++) {
				Weight[i][j][k] = (double)rand() / (double)RAND_MAX;
			}
		}
	}
	for (int i = 0; i < LAYER; i++) {
		for (int j = 0; j < NODE; j++) {
			Threshold[i][j] = (double)rand() / (double)RAND_MAX;
		}
	}
}

void BP::Forward() {
	for (int j = 0; j < hidden_num; j++) {
		Type t = 0;
		for (int i = 0; i < in_num; i++) {
			t += Weight[1][i][j] * X[0][i];
		}
		t += Threshold[1][j];
		X[1][j] = Sigmoid(t);
	}
	for (int j = 0; j < out_num; j++) {
		Type t = 0;
		for (int i = 0; i < hidden_num; i++) {
			t += Weight[2][i][j] * X[1][i];
		}
		t += Threshold[2][j];
		X[2][j] = Sigmoid(t);
	}
}

Type BP::GetError(int cnt) {
	Type ans = 0;
	for (int i = 0; i < out_num; i++) {
		ans += 0.5 * (X[2][i] - data.at(cnt).y[i]) * (X[2][i] - data.at(cnt).y[i]);
	}
	return ans;
}

void BP::Reverse(int cnt) {
	CalcDelta(cnt);
	Update();
}

Type BP::GetAccu() {
	Type ans = 0;
	int num = data.size();
	for (int i = 0; i < num; i++) {
		int m = data.at(i).x.size();
		for (int j = 0; j < m; j++) {
			X[0][j] = data.at(i).x[j];
		}
		Forward();
		int n = data.at(i).y.size();
		for (int j = 0; j < n; j++) {
			ans += 0.5 * (X[2][j] - data.at(i).y[j]) * (X[2][j] - data.at(i).y[j]);
		}
	}
	return ans / num;
}

void BP::CalcDelta(int cnt) {
	for (int i = 0; i < out_num; i++) {
		Delta[2][i] = (X[2][i] - data.at(cnt).y[i]) * X[2][i] * (A - X[2][i]) / (A * B);
	}
	for (int i = 0; i < hidden_num; i++) {
		Type t = 0;
		for (int j = 0; j < out_num; j++) {
			t += Weight[2][i][j] * Delta[2][j];
		}
		Delta[1][i] = t * X[1][i] * (A - X[1][i]) / (A * B);
	}
}

void BP::Update() {
	for (int i = 0; i < hidden_num; i++) {
		for (int j = 0; j < out_num; j++) {
			Weight[2][i][j] -= ETA_W * Delta[2][j] * X[1][i];
		}
	}
	for (int i = 0; i < out_num; i++) {
		Threshold[2][i] -= ETA_B * Delta[2][i];
	}
	for (int i = 0; i < in_num; i++) {
		for (int j = 0; j < hidden_num; j++) {
			Weight[1][i][j] -= ETA_W * Delta[1][j] * X[0][i];
		}
	}
	for (int i = 0; i < hidden_num; i++) {
		Threshold[1][i] -= ETA_B * Delta[1][i];
	}
}

Type BP::Sigmoid(const Type x) {
	return A / (1 + exp(-x / B));
}

double sample[41][4] = 
{
	{ 0,0,0,0 },
	{ 5,1,4,19.020 },
	{ 5,3,3,14.150 },
	{ 5,5,2,14.360 },
	{ 5,3,3,14.150 },
	{ 5,3,2,15.390 },
	{ 5,3,2,15.390 },
	{ 5,5,1,19.680 },
	{ 5,1,2,21.060 },
	{ 5,3,3,14.150 },
	{ 5,5,4,12.680 },
	{ 5,5,2,14.360 },
	{ 5,1,3,19.610 },
	{ 5,3,4,13.650 },
	{ 5,5,5,12.430 },
	{ 5,1,4,19.020 },
	{ 5,1,4,19.020 },
	{ 5,3,5,13.390 },
	{ 5,5,4,12.680 },
	{ 5,1,3,19.610 },
	{ 5,3,2,15.390 },
	{ 1,3,1,11.110 },
	{ 1,5,2,6.521 },
	{ 1,1,3,10.190 },
	{ 1,3,4,6.043 },
	{ 1,5,5,5.242 },
	{ 1,5,3,5.724 },
	{ 1,1,4,9.766 },
	{ 1,3,5,5.870 },
	{ 1,5,4,5.406 },
	{ 1,1,3,10.190 },
	{ 1,1,5,9.545 },
	{ 1,3,4,6.043 },
	{ 1,5,3,5.724 },
	{ 1,1,2,11.250 },
	{ 1,3,1,11.110 },
	{ 1,3,3,6.380 },
	{ 1,5,2,6.521 },
	{ 1,1,1,16.000 },
	{ 1,3,2,7.219 },
	{ 1,5,3,5.724 }
};

int main() {
	vector <Data> data;
	for (int i = 0; i < 41; i++) {
		Data t;
		for (int j = 0; j < 3; j++) {
			t.x.push_back(sample[i][j]);
		}
		t.y.push_back(sample[i][3]);
		data.push_back(t);
	}
	BP *bp = new BP();
	bp->GetData(data);
	bp->Train();
	while (true) {
		vector <Type> in;
		for (int i = 0; i < 3; i++) {
			Type v;
			scanf("%lf", &v);
			in.push_back(v);
		}
		vector <Type> out;
		out = bp->ForeCast(in);
		printf("%lf\n", out[0]);
	}
    return 0;
}

