
#include "Import.hpp"
#include "DetailedConfig.hpp"
#include "LatentModel.hpp"
#include "OrbitModel.hpp"
#include "Task.hpp"
#include <omp.h>
#include "DataModel.hpp"


/*
int main()
{
    int r = 238;
    int e = 14407;
    vector<int> a;
    a.resize(10);
    int len = a.size();
    
    
    //const int r = len;
    //const int e = len;
    //cout << r << e << endl;
    
    double ***arr3D = new double**[r];
    for (int i = 0; i<r; i++) {
        arr3D[i] = new double*[e];
        for (int j = 0; j<e; j++) {
            arr3D[i][j] = new double[e];
 
            for (int k = 0; k<e; k++) {
                arr3D[i][j][k] = (double)2.5;
            }
 
        }
    }
    vector<vector<vector<double>>> tensor(r);
    
    
    for(int i=0;i<r;i++){
        tensor[i].resize(e);
        for(int j=0;j<e;j++){
            tensor[i][j].resize(e);
        }
    }

    
    cout << tensor[3][3][2] << endl;
    
    cout<< arr3D[1345][14950][14950] << endl;
    cout<< "done"<< endl;
}
*/
/*
int main(){
    int e = 12000;
    int r = 238;
    int i, j, k;
    int*** arr = new int**[r];
    for(int i = 0; i < e; i++)
    {
        arr[i] = new int*[e];
        for(int j=0;j<e;j++){
            arr[i][j] = new int[e];
        }
        
    }
    cout<<"start memset"<<endl;
    
    for(i=0;i<r;i++)
        for(j=0;j<e;j++)
            memset(arr[i][j], 0, sizeof(int)*e);
    
    //arr[0][0][2] = 6;
    //cout << "print: "<< arr[2][2] << endl;
    for(int i = 0; i < 3; i++)
    {
        //memset(arr[i], 0, sizeof(int)*10);
        for(int j=0;j<3;j++){
            for(int k=0;k<3;k++){
                cout<<arr[i][j][k]<<" ";
            }
            cout<<endl;
        }
        cout<<endl;
        
    }
    //cout << "arr: "<< sizeof(arr) << endl;
    
    
}
*/


// 400s for each experiment.
int main(int argc, char* argv[])
{
	//omp_set_num_threads(6);

	Model* model = nullptr;
    model = new TransE(FB15KT, LinkPredictionTail, report_path, 50, 0.01, 5);
    int global = 1;
    
    int cnt_e = model->count_entity();
    int cnt_r = model->count_relation();
    int total_train = model->data_model.data_train.size();
    cout << cnt_e << endl;
    cout << cnt_r << endl;
    cout << total_train << endl;
    
    while(global<5){
        model->run(500);
        model->test();
        global++;
    }
	

    cout << "done" << endl;
    int x;
    cin >> x;
    
	return 0;
}



/*
#include <map>
#include <iostream>
#include <cassert>

int main(int argc, char **argv)
{
    std::map<pair<pair<int, int>,int>, double> m;
    
    pair<pair<int, int>,int> triplet_data;
    triplet_data.first.first = 1;
    triplet_data.first.second = 2;
    triplet_data.second = 3;
    
    m[triplet_data] = 23.0;
    // retrieve
    std::cout << m[triplet_data] << '\n';
    std::map<pair<pair<int, int>,int>, double>::iterator i = m.find(triplet_data);
    assert(i != m.end());
    std::cout << "Key: " << i->first.first.second << " Value: " << i->second << '\n';
    cout<<"after clear"<<endl;
    m.clear();
    std::cout << m[triplet_data] << '\n';
    
    pair<pair<int, int>,int> triplet_data1;
    triplet_data1.first.first = 1;
    triplet_data1.first.second = 2;
    triplet_data1.second = 34;
    
    std::cout << m[triplet_data1] << '\n';
    
    
    return 0;
}

*/
/*
#include "lrucache.hpp"

int main(){
    cache::lru_cache<tuple<int,int,int>, double> cache(3);
    //tuple<int,int,int> test(1,2,1);
    
    //cache.put(test, 2.0);
    //cache.put("two", -2.0);
    
//double from_cache = cache.get(test);
    
    //cout<<from_cache<<endl;
}

*/
/*
using namespace std;

int main()
{
    
    cache::lru_cache<std::tuple<int, int, int>, double> cache(2);
    pair<pair<int, int>, int> triplet;
    triplet.first.first = 100;
    triplet.first.second = 300;
    triplet.second = 200;
    std::tuple<int, int, int> keyT = make_tuple(triplet.first.first, triplet.second, triplet.first.second);
    std::tuple<int, int, int> key1 = make_tuple(1, 1, 1);
    std::tuple<int, int, int> key2 = make_tuple(1, 1, 2);
    std::tuple<int, int, int> key3 = make_tuple(1, 1, 3);
    std::tuple<int, int, int> key4 = make_tuple(1, 1, 4);
    
    cache.put(keyT, 11001.0);
    cache.put(key2, 112.0);

    double ret;
    if (cache.exists(keyT)) {
        ret = cache.get(keyT);
    }
    
    cout << ret << endl;
    
    int x;
    cin >> x;
}

*/

/*
int main(int argc, char* argv[])
{
    cout << "starting" << endl;
    
    Model* model = nullptr;
    //string path = "C:\\Users\\safat\\Documents\\Programming\\Embedding-master\\saved_model.txt";
    //SetPriorityClass(GetCurrentProcess(), PROCESS_MODE_BACKGROUND_BEGIN);
    
    //SetThreadPriority(GetCurrentThread(), THREAD_MODE_BACKGROUND_END);
    model = new TransE(FB15KT, LinkPredictionTail, report_path, 50, 0.01, 1);
    
    
    //model->run(50);
    
    //model->save(path);
    
    //model->test();
    
    //model->load(path);
    
    int cnt_e = model->count_entity();
    int cnt_r = model->count_relation();
    int total_train = model->data_model.data_train.size();
    cout << cnt_e << endl;
    cout << cnt_r << endl;
    //cout << total_train << endl; // 272116

     int cnt = 0;
    int i;
#pragma omp parallel for
    for (i=0; i < total_train; i++)
    {
        auto e = model->data_model.data_train[i];

        pair<pair<int, int>, int> t = e;
        //cout<<"count no: "<<cnt<<endl;
         cnt++;
         //if (cnt > 2) break;
         int head = (e).first.first;
         int tail = (e).first.second;
         int relation = (e).second;
         pair<pair<int, int>, int> triplet_f;
        
        model->energybased_false_sampling(e, triplet_f);
        //cout << "For real triplet:  h:" << head << " r: " << relation << " t: " << tail << "------->";
        //cout << "Find false triplet: h:" << triplet_f.first.first << " r: " << triplet_f.second << " t: " << triplet_f.first.second << endl;
        
    
     }
    
    //cout<<"count no: "<<cnt<<endl;
    cout << "done" << endl;
    int x;
    cin >> x;
    //model->test();
    return 0;
}
*/
