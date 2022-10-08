#include <iostream>
#include<cmath>
#include<math.h>
#include<vector>
#include<iomanip>
using namespace std;
class data {
protected:
    double  x_table[20][3] = { { 1, 8, 7 }, { 2, 8, 7 }, { 0, 8, 7 }, { 1, 5, 6 }, { 1, 9, 7 }, { 1, 8, 10 }, { 1, 8, 6 }, { 2, 8, 6 }, { 2, 8, 1 }, { 0, 9, 3}, { 0, 8, 9 }, { 0, 5, 7 }, { 0, 9, 7 }, { 1, 8, 9 }, { 1, 7, 8 }, { 2, 9, 7 }, { 2, 9, 10 }, { 2, 6, 7 }, { 3, 8, 6 }, { 1, 8, 5 }};
    double x1_table[20][3];
    double d1_table[20];
    double d11[20];
    double d2_table[20];
    long double navch_vybir[16][5];
    long double control_vybir[4][5];
    double AVG;
    const long double eta = 0.3;
    double minX1;
    double minX2;
    double minX3;
    double maxX1;
    double maxX2;
    double maxX3;
    double minD1, maxD1;
    double q;
    int index_mass[20];
    double function(double x1, double x2, double x3)
    {
        return pow(x1,3) + sqrt(x2) - pow(x3,4);
       // return sin(x1) + tan(x2) - tan(x3);
    }
    short get_d2(double d1)
    {
        if (d1 > AVG)
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }
    double get_AVG()
    {
        double avg = 0;
        for (int i = 0; i < 20; i++)
        {
            avg += d1_table[i];
        }
        avg = avg / 20;
        AVG = avg;
        return AVG;
    }
    long double activation_func(long double x)
    {
        return (1 / (1 + exp(-1.0 * x)));
    }
    long double derivative(long double x)
    {
        return activation_func(x) * (1 - activation_func(x));
    }
    data get_data()
    {
        for (int i = 0; i < 20; i++)
        {
            x1_table[i][0] = x_table[i][0];
            x1_table[i][1] = x_table[i][1];
            x1_table[i][2] = x_table[i][2];
        }
        for (int i = 0; i < 20; i++)
        {
            d1_table[i] = function(x_table[i][0], x_table[i][1], x_table[i][2]);
            d11[i] = d1_table[i];
        }
        AVG = get_AVG();
        for (int i = 0; i < 20; i++)
        {
            d2_table[i] = get_d2(d1_table[i]);
        }
        minX1 = 0;
        minX2 = 7;
        minX3 = 6;
        maxX1 = 2;
        maxX2 = 9;
        maxX3 = 8;
        for (int i = 0; i < 20; i++)
        {
            //нормирование x1
            x_table[i][0] = (x_table[i][0] - minX1) / (maxX1 - minX1);
            //нормирование x2
            x_table[i][1] = (x_table[i][1] - minX2) / (maxX2 - minX2);
            //нормирование x3
            x_table[i][2] = (x_table[i][2] - minX2) / (maxX3 - minX3);
        }
        //нормирование d1
        minD1 = d1_table[0];
        for (int i = 0; i < 20; ++i)//ищем минимальное значение заданой функции 
        {
            if (d1_table[i] < minD1)
            {
                minD1 = d1_table[i];
            }
        }
        maxD1 = d1_table[0];
        for (int i = 0; i < 20; ++i)//ищем максимальное значение функции
        {
            if (d1_table[i] > maxD1)
            {
                maxD1 = d1_table[i];
            }
        }
        for (int i = 0; i < 20; i++)//нормируем
        {
            d1_table[i] = (d1_table[i] - minD1) / (maxD1 - minD1);
        }
        //заполнение контрольной и учебной выборки рандомными элементами 
        int rnd;
        bool flag = true, flag1 = true;
        int tmp = 0;
        for (int i = 0; i < 20; i++)
        {
            while (flag1)
            {
                rnd = rand() % (19 - 0 + 1) + 0; 
                while (tmp < i && flag)
                {
                    if (rnd == index_mass[tmp])
                    {
                        flag = false;
                    }
                    tmp++;
                }
                if (!flag)
                {
                    tmp = 0;
                    flag = true;
                }
                else
                {
                    flag1 = false;
                }
            }
            if (!flag1)
            {
                flag1 = true;
                index_mass[i] = rnd;
                tmp = 0;
                if (i > 15)
                {
                    control_vybir[i - 16][0] = x_table[rnd][0];
                    control_vybir[i - 16][1] = x_table[rnd][1];
                    control_vybir[i - 16][2] = x_table[rnd][2];
                    control_vybir[i - 16][3] = d1_table[rnd];
                    control_vybir[i - 16][4] = d2_table[rnd];
                }
                else
                {
                    navch_vybir[i][0] = x_table[rnd][0];
                    navch_vybir[i][1] = x_table[rnd][1];
                    navch_vybir[i][2] = x_table[rnd][2];
                    navch_vybir[i][3] = d1_table[rnd];
                    navch_vybir[i][4] = d2_table[rnd];
                }
            }
        }
        return *this;
    }
    double de_norm(double q1)
    {
        q = (q1 * (maxD1 - minD1)) + (minD1);
        return q;
    }
};
class NW : public data {
private:
    //weights 
    long double input_layer1[3];
    long double input_layer2[3];
    long double input_layer3[3];
    long double hiden_layer1[2];
    long double hiden_layer2[2];
    long double hiden_layer3[2];
    // results
    long double result_hidenLayer1;
    long double result_hidenLayer2;
    long double result_hidenLayer3;
    long double result_outputLayer1;
    long double result_outputLayer2;
    //error
    long double error_input1;
    long double error_input2;
    long double error_input3;
    long double error_hiden1;
    long double error_hiden2;
    long double error_hiden3;
    long double error_output1;
    long double error_output2;
public:
    NW set_weights()
    {
        for (int i = 0; i < 3; i++)
        {
            input_layer1[i] = -1 + double(rand()) / RAND_MAX * (1 + 1);
            input_layer2[i] = -1 + double(rand()) / RAND_MAX * (1 + 1);
            input_layer3[i] = -1 + double(rand()) / RAND_MAX * (1 + 1);
        }
        for (int i = 0; i < 2; i++)
        {
            hiden_layer1[i] = -1 + double(rand()) / RAND_MAX * (1 + 1);
            hiden_layer2[i] = -1 + double(rand()) / RAND_MAX * (1 + 1);
            hiden_layer3[i] = -1 + double(rand()) / RAND_MAX * (1 + 1);
        }
        return *this;
    }
    NW get_result(int n, short q)
    {
        double sum_error = 0;
        if (q == 0)
        {
            cout << "X1 " << "  X2" << "  X3" << "  T1" << "  Y1" << "  T2" << " Y2" << endl;
        }
        //hiden_result
        for (int i = 0; i < n; i++)
        {
            result_hidenLayer1 = activation_func(input_layer1[0] * navch_vybir[i][0] + input_layer2[0] * navch_vybir[i][1] + input_layer3[0] * navch_vybir[i][2]);
            result_hidenLayer2 = activation_func(input_layer1[1] * navch_vybir[i][0] + input_layer2[1] * navch_vybir[i][1] + input_layer3[1] * navch_vybir[i][2]);
            result_hidenLayer3 = activation_func(input_layer1[2] * navch_vybir[i][0] + input_layer2[2] * navch_vybir[i][1] + input_layer3[2] * navch_vybir[i][2]);
            //output_result
            result_outputLayer1 = activation_func(hiden_layer1[0] * result_hidenLayer1 + hiden_layer2[0] * result_hidenLayer2 + hiden_layer3[0] * result_hidenLayer3);
            result_outputLayer2 = activation_func(hiden_layer1[1] * result_hidenLayer1 + hiden_layer2[1] * result_hidenLayer2 + hiden_layer3[1] * result_hidenLayer3);
            //error
            error_output1 = navch_vybir[i][3] - result_outputLayer1;
            error_output2 = navch_vybir[i][4] - result_outputLayer2;
            error_hiden1 = error_output1 * hiden_layer1[0] + error_output2 * hiden_layer1[1];
            error_hiden2 = error_output1 * hiden_layer2[0] + error_output2 * hiden_layer2[1];
            error_hiden3 = error_output1 * hiden_layer3[0] + error_output2 * hiden_layer3[1];
            if (q == 0)
            {
                cout << navch_vybir[i][0] << "  " << navch_vybir[i][1] << "  " << navch_vybir[i][2] << "  " << navch_vybir[i][3] << "  " << result_outputLayer1 << "  " << navch_vybir[i][4] << "  " << result_outputLayer2 << endl;
                sum_error += pow(navch_vybir[i][3] - result_outputLayer1, 2);
                sum_error += pow(navch_vybir[i][4] - result_outputLayer2, 2);
            }
        }
        if (q == 0)
        {
            cout << sum_error / 16 << endl;
        }
        return *this;
    }
    NW train()
    {
       int time = clock();
       for (int epoch = 0; epoch < 100000; epoch++)
       {
           for(int i = 16; i > -1; i--)
           {
                 get_result(i,1);
                //input layer
                input_layer1[0] = input_layer1[0] + eta * error_hiden1 * derivative(input_layer1[0] * navch_vybir[i][0] + input_layer2[0] * navch_vybir[i][1] + input_layer3[0] * navch_vybir[i][2]) * navch_vybir[i][0];
                input_layer2[0] = input_layer2[0] + eta * error_hiden1 * derivative(input_layer1[0] * navch_vybir[i][0] + input_layer2[0] * navch_vybir[i][1] + input_layer3[0] * navch_vybir[i][2]) * navch_vybir[i][1];
                input_layer3[0] = input_layer3[0] + eta * error_hiden1 * derivative(input_layer1[0] * navch_vybir[i][0] + input_layer2[0] * navch_vybir[i][1] + input_layer3[0] * navch_vybir[i][2]) * navch_vybir[i][2];
                input_layer1[1] = input_layer1[1] + eta * error_hiden2 * derivative(input_layer1[1] * navch_vybir[i][0] + input_layer2[1] * navch_vybir[i][1] + input_layer3[1] * navch_vybir[i][2]) * navch_vybir[i][0];
                input_layer2[1] = input_layer2[1] + eta * error_hiden2 * derivative(input_layer1[1] * navch_vybir[i][0] + input_layer2[1] * navch_vybir[i][1] + input_layer3[1] * navch_vybir[i][2]) * navch_vybir[i][1];
                input_layer3[1] = input_layer3[1] + eta * error_hiden2 * derivative(input_layer1[1] * navch_vybir[i][0] + input_layer2[1] * navch_vybir[i][1] + input_layer3[1] * navch_vybir[i][2]) * navch_vybir[i][2];
                input_layer1[2] = input_layer1[2] + eta * error_hiden3 * derivative(input_layer1[2] * navch_vybir[i][0] + input_layer2[2] * navch_vybir[i][1] + input_layer3[2] * navch_vybir[i][2]) * navch_vybir[i][0];
                input_layer2[2] = input_layer2[2] + eta * error_hiden3 * derivative(input_layer1[2] * navch_vybir[i][0] + input_layer2[2] * navch_vybir[i][1] + input_layer3[2] * navch_vybir[i][2]) * navch_vybir[i][1];
                input_layer3[2] = input_layer3[2] + eta * error_hiden3 * derivative(input_layer1[2] * navch_vybir[i][0] + input_layer2[2] * navch_vybir[i][1] + input_layer3[2] * navch_vybir[i][2]) * navch_vybir[i][2];
                //hiden layer
                hiden_layer1[0] = hiden_layer1[0] + eta * error_output1 * derivative(hiden_layer1[0] * result_hidenLayer1 + hiden_layer2[0] * result_hidenLayer2 + hiden_layer3[0] * result_hidenLayer3) * result_hidenLayer1;
                hiden_layer2[0] = hiden_layer2[0] + eta * error_output1 * derivative(hiden_layer1[0] * result_hidenLayer1 + hiden_layer2[0] * result_hidenLayer2 + hiden_layer3[0] * result_hidenLayer3) * result_hidenLayer2;
                hiden_layer3[0] = hiden_layer3[0] + eta * error_output1 * derivative(hiden_layer1[0] * result_hidenLayer1 + hiden_layer2[0] * result_hidenLayer2 + hiden_layer3[0] * result_hidenLayer3) * result_hidenLayer3;
                hiden_layer1[1] = hiden_layer1[1] + eta * error_output2 * derivative(hiden_layer1[1] * result_hidenLayer1 + hiden_layer2[1] * result_hidenLayer2 + hiden_layer3[1] * result_hidenLayer3) * result_hidenLayer1;
                hiden_layer2[1] = hiden_layer2[1] + eta * error_output2 * derivative(hiden_layer1[1] * result_hidenLayer1 + hiden_layer2[1] * result_hidenLayer2 + hiden_layer3[1] * result_hidenLayer3) * result_hidenLayer2;
                hiden_layer3[1] = hiden_layer3[1] + eta * error_output2 * derivative(hiden_layer1[1] * result_hidenLayer1 + hiden_layer2[1] * result_hidenLayer2 + hiden_layer3[1] * result_hidenLayer3) * result_hidenLayer3;
           }
       }
        cout << "Time = " << clock() - time << endl;
        return *this;
    }
    NW control()
    {
        //hiden_result
        double sum_error = 0;
        for (int i = 0; i < 4; i++)
        {
            result_hidenLayer1 = activation_func(this->input_layer1[0] * control_vybir[i][0] + this->input_layer2[0] * control_vybir[i][1] + this->input_layer3[0] * control_vybir[i][2]);
            result_hidenLayer2 = activation_func(this->input_layer1[1] * control_vybir[i][0] + this->input_layer2[1] * control_vybir[i][1] + this->input_layer3[1] * control_vybir[i][2]);
            result_hidenLayer3 = activation_func(this->input_layer1[2] * control_vybir[i][0] + this->input_layer2[2] * control_vybir[i][1] + this->input_layer3[2] * control_vybir[i][2]);
            result_outputLayer1 = activation_func(this->hiden_layer1[0] * result_hidenLayer1 + hiden_layer2[0] * result_hidenLayer2 + hiden_layer3[0] * result_hidenLayer3);
            result_outputLayer2 = activation_func(this->hiden_layer1[1] * result_hidenLayer1 + hiden_layer2[1] * result_hidenLayer2 + hiden_layer3[1] * result_hidenLayer3);
            cout << control_vybir[i][0] << "  " << control_vybir[i][1] << "  " << control_vybir[i][2] << "  " << control_vybir[i][3] << "  " << result_outputLayer1 << "  " << control_vybir[i][4] << "  " << result_outputLayer2 << endl;
            sum_error += pow(control_vybir[i][3] - result_outputLayer1, 2);
            sum_error += pow(control_vybir[i][4] - result_outputLayer2, 2);
        }
        cout << sum_error / 4 << endl;

        for (int i = 0; i < 4; i++)
        {
            result_hidenLayer1 = activation_func(this->input_layer1[0] * control_vybir[i][0] + this->input_layer2[0] * control_vybir[i][1] + this->input_layer3[0] * control_vybir[i][2]);
            result_hidenLayer2 = activation_func(this->input_layer1[1] * control_vybir[i][0] + this->input_layer2[1] * control_vybir[i][1] + this->input_layer3[1] * control_vybir[i][2]);
            result_hidenLayer3 = activation_func(this->input_layer1[2] * control_vybir[i][0] + this->input_layer2[2] * control_vybir[i][1] + this->input_layer3[2] * control_vybir[i][2]);
            result_outputLayer1 = activation_func(this->hiden_layer1[0] * result_hidenLayer1 + hiden_layer2[0] * result_hidenLayer2 + hiden_layer3[0] * result_hidenLayer3);
            result_outputLayer2 = activation_func(this->hiden_layer1[1] * result_hidenLayer1 + hiden_layer2[1] * result_hidenLayer2 + hiden_layer3[1] * result_hidenLayer3);
            cout << x1_table[index_mass[i+15]][0] << "  " << x1_table[index_mass[i+15]] [1] << "  " << x1_table[index_mass[i+15]][2] << "  " << d11[index_mass[i+15]] << "  " << this->de_norm(result_outputLayer1) << "  " << control_vybir[i][4] << "  " << result_outputLayer2 << endl;
        }
        
        return *this;
    }
    NW interface()
    {
        short a;
        bool flag = true;
        this->get_data();
        this->set_weights();

        while (flag)
        {
            cout << "Input what you want to do " << endl;
            cout << "1 - run train data, 2 - train, 3 - run test data, 4 - exit" << endl;
            cin >> a;
            switch (a)
            {
            case 1:
                this->get_result(16, 0);
                break;
            case 2:
                this->train();
                break;
            case 3:
                this->control();
                break;
            case 4:
                flag = false;
                break;
            }
        }
        return *this;
    }
};
int main()
{
    NW a;
    a.interface();
    return 0;
}
