#pragma once
#include "Import.hpp"
#include "ModelConfig.hpp"
#include "DataModel.hpp"
#include <boost/progress.hpp>
#include <map>
#include "lrucache.hpp"

using namespace std;
using namespace arma;



class Model
{
public:
	const DataModel&	data_model;
	const TaskType		task_type;
	const bool			be_deleted_data_model;
    
public:
	ModelLogging&		logging;

public:
	int	epos;
    double prb_rndm;
    int len_set_e;
    int len_set_r;
    double sampling_percent = 0.1; // [between 0 to 1]
    int indx = 0;
    int sampling_number;
    cache::lru_cache<std::tuple<int, int, int>, double> cache;

public:
	Model(const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path)
		:data_model(*(new DataModel(dataset))), task_type(task_type),
		logging(*(new ModelLogging(logging_base_path))),
		be_deleted_data_model(true), cache(100000)
	{
		epos = 0;
		best_triplet_result = 0;
        init_false_sample();
		std::cout << "Ready" << endl;

		logging.record() << "\t[Dataset]\t" << dataset.name;
		logging.record() << TaskTypeName(task_type);
	}

	Model(const Dataset& dataset,
		const string& file_zero_shot,
		const TaskType& task_type,
		const string& logging_base_path)
		:data_model(*(new DataModel(dataset))), task_type(task_type),
		logging(*(new ModelLogging(logging_base_path))),
		be_deleted_data_model(true), cache(100000)
	{
		epos = 0;
		best_triplet_result = 0;
        //init_false_sample();
		std::cout << "Ready" << endl;

		logging.record() << "\t[Dataset]\t" << dataset.name;
		logging.record() << TaskTypeName(task_type);
	}

	Model(const DataModel* data_model,
		const TaskType& task_type,
		ModelLogging* logging)
		:data_model(*data_model), logging(*logging), task_type(task_type),
		be_deleted_data_model(false), cache(100000)
	{
		epos = 0;
		best_triplet_result = 0;
        //init_false_sample();
	}

public:
	virtual double prob_triplets(const pair<pair<int, int>, int>& triplet) = 0;
	virtual void train_triplet(const pair<pair<int, int>, int>& triplet) = 0;
    //virtual void train_triplet1(const pair<pair<int, int>, int>& triplet) = 0;
    
    
public:
    virtual void train(bool last_time = false) // original is: virtual void train(bool last_time = false)
	{
		++epos;
        int len = data_model.data_train.size();
        int i = 0;
        //cout << "len: "<< len << endl;
#pragma omp parallel for
        for (i = 0; i < len; i++)
        {
            auto e = &data_model.data_train[i];
            train_triplet(*e);
        }
        
	}

    void run(int total_epos)
    {
        logging.record() << "\t[Epos]\t" << total_epos;
        
        --total_epos;
        boost::progress_display    cons_bar(total_epos);
        while (total_epos-- > 0)
        {
            //cout << "total_epos: "<< total_epos << endl;
            ++cons_bar;
            train();
            
            if (task_type == TripletClassification)
                test_triplet_classification();
        }
        
        train(true);
    }

public:
	double		best_triplet_result;
	double		best_link_mean;
	double		best_link_hitatten;
	double		best_link_fmean;
	double		best_link_fhitatten;

	void reset()
	{
		best_triplet_result = 0;
		best_link_mean = 1e10;
		best_link_hitatten = 0;
		best_link_fmean = 1e10;
		best_link_fhitatten = 0;
	}

	void test(int hit_rank = 10)
	{
		logging.record();

		best_link_mean = 1e10;
		best_link_hitatten = 0;
		best_link_fmean = 1e10;
		best_link_fhitatten = 0;

		if (task_type == LinkPredictionHead || task_type == LinkPredictionTail || task_type == LinkPredictionRelation)
			test_link_prediction(hit_rank);
		if (task_type == LinkPredictionHeadZeroShot || task_type == LinkPredictionTailZeroShot || task_type == LinkPredictionRelationZeroShot)
			test_link_prediction_zeroshot(hit_rank);
		else
			test_triplet_classification();
        //m.clear();
	}
   

    void init_false_sample(){
        len_set_e = data_model.set_entity.size();
        len_set_r = data_model.set_relation.size();
        sampling_number = sampling_percent * len_set_e;
        prb_rndm = (rand() % 11) / 10;
     
    }
    
    template<class TupType, size_t... I>
    void print(const TupType& _tup, std::index_sequence<I...>)
    {
        std::cout << "(";
        (..., (std::cout << (I == 0? "" : ", ") << std::get<I>(_tup)));
        std::cout << ")\n";
    }
    
    template<class... T>
    void print (const std::tuple<T...>& _tup)
    {
        print(_tup, std::make_index_sequence<sizeof...(T)>());
    }
    
    
public:
    void energybased_false_sampling(
                                      const pair<pair<int, int>, int>& origin,
                                      pair<pair<int, int>, int>& triplet)
    {
        triplet = origin;
        int head = origin.first.first;
        int tail = origin.first.second;
        int relation = origin.second;
        int false_tail;
        vector<int> entity_no;
        vector<double> entity_enrg;
        std::tuple<int, int, int> key;
        
        // For this implementation: true triplets will have energies close to zero
        // and false triplets will have larger negative values.
        
        int i = 0;
        double mx  = -1000000;
        double mn = 1;
        
        double curr_enrg = 0;
        double value;
        
        //cout<< "init mx: "<< mx << endl;
        //cout << "init mn: "<< mn << endl;
        
        int largest_en_i;
        
        
        for(i=0; i<sampling_number; i++){
            
            false_tail = rand() % len_set_e;
            if (false_tail == tail)
                continue;
            
            triplet.first.second = false_tail;
            key = make_tuple(triplet.first.first, triplet.second, triplet.first.second);
            //cout << "---" << endl;
            //print(key);
            
#pragma omp critical
            {
            
            if (!cache.exists(key)) {
                //cout << cache.exists(key) << endl;
                value = prob_triplets(triplet);
                cache.put(key, value);
                //cout << cache.exists(key) << endl;
            }
            
            //cout<< indx << endl;
            //cout << cache.exists(key) << endl;
            //print(key);
            curr_enrg = cache.get(key);
            entity_no.push_back(false_tail);
            entity_enrg.push_back(curr_enrg);
                
            }
            
        
            //cout<<"final: "<< m[triplet] << endl;
            if(curr_enrg > mx) mx = curr_enrg;
            if(curr_enrg < mn)
            {
                mn = curr_enrg;
                //largest_en_i = false_tail;
            }
        
            
            }
        
        
        //cout<< "final mx: "<< mx << endl;
        //cout << "final mn: "<< mn << endl;
        
        // for sampling largest negative value
        
        //triplet.first.second = largest_en_i;
        //cout << "curr: " << tensor_energy[relation][head][largest_en_i] << endl;
        //return;
        

        
        double v = (mn + mx)/2.0;
        int ln = entity_no.size();
        
        //cout << "v: "<< v <<endl;
        
        for(i=0; i<ln; i++){
            
            false_tail = entity_no[i];
            triplet.first.second = false_tail;
            key = make_tuple(triplet.first.first, triplet.second, triplet.first.second);
            //curr_enrg = cache.get(key);
            
            if(entity_enrg[i] <= v)
            {
                //cout << "curr: " << curr_enrg << endl;
                return;
            }
            
        }
        
    }
    /*
    
public:
    void probabilitybased_false_sampling(
                                         const pair<pair<int, int>, int>& origin,
                                         pair<pair<int, int>, int>& triplet) 
    {
        triplet = origin;
        int head = origin.first.first;
        int tail = origin.first.second;
        int relation = origin.second;
        int false_tail;
        
        vector<double> ener; // energy vector
        vector<int> entity_no;
        double total_energy = 0;
        //cout<< "sampling number: " << sampling_number<<endl;
        //#pragma omp parallel for
        for (int i = 0; i < sampling_number; i++) {
            false_tail = rand() % len_set_e;
            if (false_tail == tail)
                continue;
            
            triplet.first.second = false_tail;
            
            if(!(tensor_energy[relation][head][false_tail] < 0))
            {
                tensor_energy[relation][head][false_tail] = prob_triplets(triplet);
            }
            total_energy += tensor_energy[relation][head][false_tail];
            ener.push_back(tensor_energy[relation][head][false_tail]);
            entity_no.push_back(false_tail);
            
            
            //cout << i << " current energy: " << ener[i] << endl;
            //total_energy += ener[i];
        }
        //cout << "sample Energy: " << total_energy << endl;
        //prb_rndm = 1.0 / abs(total_energy);
        //cout<< "sample_prob: " << prb_rndm << endl;
        //double temp = (total_energy * (1/sampling_percent));
        //cout << "Total Approximate Energy: " << temp << endl;
        //cout << "Appro. Probability: " << 1.0 / temp <<endl;
        
        prb_rndm = (rand() % 11) / 10.0;
        std::discrete_distribution<int> test(ener.begin(), ener.end());
         
        int indx = 0;
        double prev_prob = 0;
        double cum_sum = 0;
        //cout<<"random: " <<prb_rndm <<endl;
        for (double x : test.probabilities()) {
             cum_sum += x;
             
             if(prb_rndm<cum_sum){
                 triplet.first.second = entity_no[indx];
                 //cout<<"origin: " << prob_triplets(origin) <<" false: "<< prob_triplets(triplet) <<endl;
                 return;
             //cum_prb[indx] = prev_prob + x;
             //prev_prob = cum_prb[indx];
             //std::cout << x << " " << indx << " cum_sum " << cum_sum << " cum_prob "<< cum_prb[indx] <<endl;
             }
             indx++;
        }
        
        triplet.first.second = entity_no[(indx-1)];
        return;
     
        
        /*
        rndm = (rand() % 11) / 10.0;
        bool go = true;
        double crnt_prb;
        double prev_prob = 0;
        while (go)
        {
            //cout << "Genereated random number is: " << rndm << endl;
            prev_prob = 0;
            for (auto i = 0; i < len_e; i++) {
                if (i == tail) {
                    if (i != 0) cum_prb[i] = cum_prb[i - 1];
                    else cum_prb[0] = 0;
                    continue;
                }
                
                crnt_prb = (ener[i] / total_energy);
                cum_prb[i] = prev_prob + crnt_prb;
                prev_prob = cum_prb[i];
                
                //cout << i << " current prob "<<crnt_prb<<" cumulative sum is: " << cum_prb[i] << " rndm: "<<rndm<<endl;
                if (cum_prb[i] >= rndm) {
                    go = false;
                    int prev = i - 1;
                    //cout << "FIND i " << i << " prev " << prev << " tail " << tail << endl;
                    if (prev >= 0 && prev != tail) triplet.first.second = prev;
                    else if ((prev - 1) >= 0) triplet.first.second = prev;
                    else {
                        rndm = (rand() % 11) / 10.0;
                        go = true;
                    }
                    break;
                }
                
            }
            if (rndm == 1)
            {
                triplet.first.second = (len_e - 1);
                go = false;
            }
            //cout << cum_prb[len_e-1] << endl;
        }
     
    }
*/


public:
	void test_triplet_classification()
	{
		double real_hit = 0;
		for (auto r = 0; r < data_model.set_relation.size(); ++r)
		{
			vector<pair<double, bool>>	threshold_dev;
			for (auto i = data_model.data_dev_true.begin(); i != data_model.data_dev_true.end(); ++i)
			{
				if (i->second != r)
					continue;

				threshold_dev.push_back(make_pair(prob_triplets(*i), true));
			}
			for (auto i = data_model.data_dev_false.begin(); i != data_model.data_dev_false.end(); ++i)
			{
				if (i->second != r)
					continue;

				threshold_dev.push_back(make_pair(prob_triplets(*i), false));
			}

			sort(threshold_dev.begin(), threshold_dev.end());

			double threshold;
			double vari_mark = 0;
			int total = 0;
			int hit = 0;
			for (auto i = threshold_dev.begin(); i != threshold_dev.end(); ++i)
			{
				if (i->second == false)
					++hit;
				++total;

				if (vari_mark <= 2 * hit - total + data_model.data_dev_true.size())
				{
					vari_mark = 2 * hit - total + data_model.data_dev_true.size();
					threshold = i->first;
				}
			}

			double lreal_hit = 0;
			double lreal_total = 0;
			for (auto i = data_model.data_test_true.begin(); i != data_model.data_test_true.end(); ++i)
			{
				if (i->second != r)
					continue;

				++lreal_total;
				if (prob_triplets(*i) > threshold)
					++real_hit, ++lreal_hit;
			}

			for (auto i = data_model.data_test_false.begin(); i != data_model.data_test_false.end(); ++i)
			{
				if (i->second != r)
					continue;

				++lreal_total;
				if (prob_triplets(*i) <= threshold)
					++real_hit, ++lreal_hit;
			}

			//logging.record()<<data_model.relation_id_to_name.at(r)<<"\t"
			//	<<lreal_hit/lreal_total;
		}
        
        best_triplet_result = max(
                                  best_triplet_result,
                                  real_hit / (data_model.data_test_true.size() + data_model.data_test_false.size()));

        double accuracy = (real_hit / (data_model.data_test_true.size() + data_model.data_test_false.size()));
        logging.record() << "epos = "<< epos;
        logging.record() << "Accuracy = "<< accuracy;
        logging.record();
        logging.record() << "Best = " << best_triplet_result;
        logging.record();
        
        
		std::cout << epos << "\t Accuracy = "
			<< real_hit / (data_model.data_test_true.size() + data_model.data_test_false.size());
		
		std::cout << ", Best = " << best_triplet_result << endl;

        //logging.record() << epos<< "\t Accuracy = "<< real_hit / (data_model.data_test_true.size() + data_model.data_test_false.size())<< ", Best = " << best_triplet_result;
        
        cout<<"After Finished"<<endl;

		std::cout.flush();
	}

	void test_link_prediction(int hit_rank = 10, const int part = 0)
	{
		double mean = 0;
		double hits = 0;
		double fmean = 0;
		double fhits = 0;
		double rmrr = 0;
		double fmrr = 0;
		double total = data_model.data_test_true.size();

		double arr_mean[20] = { 0 };
		double arr_total[5] = { 0 };

		for (auto i = data_model.data_test_true.begin(); i != data_model.data_test_true.end(); ++i)
		{
			++arr_total[data_model.relation_type[i->second]];
		}

		int cnt = 0;

		boost::progress_display cons_bar(data_model.data_test_true.size() / 100);
        int i =0;
        int len = data_model.data_test_true.size();

#pragma omp parallel for
		for (i=0; i < len; i++)
		{
			++cnt;
			if (cnt % 100 == 0)
			{
				++cons_bar;
			}
            auto e = &data_model.data_test_true[i];
            
			pair<pair<int, int>, int> t = *e;
            
			int frmean = 0;
			int rmean = 0;
			double score_i = prob_triplets(*e);

			if (task_type == LinkPredictionRelation || part == 2)
			{
				for (auto j = 0; j != data_model.set_relation.size(); ++j)
				{
					t.second = j;

					if (score_i >= prob_triplets(t))
						continue;

					++rmean;

					if (data_model.check_data_all.find(t) == data_model.check_data_all.end())
						++frmean;
				}
			}
			else
			{
				for (auto j = 0; j != data_model.set_entity.size(); ++j)
				{
					if (task_type == LinkPredictionHead || part == 1)
						t.first.first = j;
					else
						t.first.second = j;

					if (score_i >= prob_triplets(t))
						continue;

					++rmean;

					if (data_model.check_data_all.find(t) == data_model.check_data_all.end())
					{
						++frmean;
						//if (frmean > hit_rank)
						//	break;
					}
				}
			}

#pragma omp critical
			{
				if (frmean < hit_rank)
					++arr_mean[data_model.relation_type[e->second]];

				mean += rmean;
				fmean += frmean;
				rmrr += 1.0 / (rmean + 1);
				fmrr += 1.0 / (frmean + 1);
                
                int head = e->first.first;
                int tail = e->first.second;
                int relation = e->second;
                
                //cout<<"rmean: "<<rmean<<" h: "<<head<<" r: "<<relation<<" t: "<<tail<<endl;
				if (rmean < hit_rank)
					++hits;
				if (frmean < hit_rank)
					++fhits;
			}
		}

		std::cout << endl;
		for (auto i = 1; i <= 4; ++i)
		{
			std::cout << i << ':' << arr_mean[i] / arr_total[i] << endl;
			logging.record() << i << ':' << arr_mean[i] / arr_total[i];
		}
		logging.record();

		best_link_mean = min(best_link_mean, mean / total);
		best_link_hitatten = max(best_link_hitatten, hits / total);
		best_link_fmean = min(best_link_fmean, fmean / total);
		best_link_fhitatten = max(best_link_fhitatten, fhits / total);

		std::cout << "Raw.BestMEANS = " << best_link_mean << endl;
		std::cout << "Raw.BestMRR = " << rmrr / total << endl;
		std::cout << "Raw.BestHITS = " << best_link_hitatten << endl;
		logging.record() << "Raw.BestMEANS = " << best_link_mean;
		logging.record() << "Raw.BestMRR = " << rmrr / total;
		logging.record() << "Raw.BestHITS = " << best_link_hitatten;

		std::cout << "Filter.BestMEANS = " << best_link_fmean << endl;
		std::cout << "Filter.BestMRR= " << fmrr / total << endl;
		std::cout << "Filter.BestHITS = " << best_link_fhitatten << endl;
		logging.record() << "Filter.BestMEANS = " << best_link_fmean;
		logging.record() << "Filter.BestMRR= " << fmrr / total;
		logging.record() << "Filter.BestHITS = " << best_link_fhitatten;
        logging.record() << "-------------------------";

		std::cout.flush();
	}

public:
	void test_link_prediction_zeroshot(int hit_rank = 10, const int part = 0)
	{
		reset();

		double mean = 0;
		double hits = 0;
		double fmean = 0;
		double fhits = 0;
		double total = data_model.data_test_true.size();

		double arr_mean[20] = { 0 };
		double arr_total[5] = { 0 };

		cout << endl;

		for (auto i = data_model.data_test_true.begin(); i != data_model.data_test_true.end(); ++i)
		{
			if (i->first.first >= data_model.zeroshot_pointer
				&& i->first.second >= data_model.zeroshot_pointer)
			{
				++arr_total[3];
			}
			else if (i->first.first < data_model.zeroshot_pointer
				&& i->first.second >= data_model.zeroshot_pointer)
			{
				++arr_total[2];
			}
			else if (i->first.first >= data_model.zeroshot_pointer
				&& i->first.second < data_model.zeroshot_pointer)
			{
				++arr_total[1];
			}
			else
			{
				++arr_total[0];
			}
		}

		cout << "0 holds " << arr_total[0] << endl;
		cout << "1 holds " << arr_total[1] << endl;
		cout << "2 holds " << arr_total[2] << endl;
		cout << "3 holds " << arr_total[3] << endl;

		int cnt = 0;
        int len = data_model.data_test_true.size();
        int i = 0;

#pragma omp parallel for
		for (i = 0; i < len; i++)
		{
			++cnt;
			if (cnt % 100 == 0)
			{
				std::cout << cnt << ',';
				std::cout.flush();
			}
            
            auto e = &data_model.data_test_true[i];
			pair<pair<int, int>, int> t = *e;
			int frmean = 0;
			int rmean = 0;
			double score_i = prob_triplets(*e);

			if (task_type == LinkPredictionRelationZeroShot || part == 2)
			{
				for (auto j = 0; j != data_model.set_relation.size(); ++j)
				{
					t.second = j;

					if (score_i >= prob_triplets(t))
						continue;

					++rmean;

					if (data_model.check_data_all.find(t) == data_model.check_data_all.end())
						++frmean;
				}
			}
			else
			{
				for (auto j = 0; j != data_model.set_entity.size(); ++j)
				{
					if (task_type == LinkPredictionHeadZeroShot || part == 1)
						t.first.first = j;
					else
						t.first.second = j;

					if (score_i >= prob_triplets(t))
						continue;

					++rmean;

					if (data_model.check_data_all.find(t) == data_model.check_data_all.end())
					{
						++frmean;
					}
				}
			}

#pragma omp critical
			{
				if (frmean < hit_rank)
				{
					if (e->first.first >= data_model.zeroshot_pointer
						&& e->first.second >= data_model.zeroshot_pointer)
					{
						++arr_mean[3];
					}
					else if (e->first.first < data_model.zeroshot_pointer
						&& e->first.second >= data_model.zeroshot_pointer)
					{
						++arr_mean[2];
					}
					else if (e->first.first >= data_model.zeroshot_pointer
						&& e->first.second < data_model.zeroshot_pointer)
					{
						++arr_mean[1];
					}
					else
					{
						++arr_mean[0];
					}
				}

				mean += rmean;
				fmean += frmean;
				if (rmean < hit_rank)
					++hits;
				if (frmean < hit_rank)
					++fhits;
			}
		}

		std::cout << endl;
		for (auto i = 0; i < 4; ++i)
		{
			std::cout << i << ':' << arr_mean[i] / arr_total[i] << endl;
			logging.record() << i << ':' << arr_mean[i] / arr_total[i];
		}
		logging.record();

		best_link_mean = min(best_link_mean, mean / total);
		best_link_hitatten = max(best_link_hitatten, hits / total);
		best_link_fmean = min(best_link_fmean, fmean / total);
		best_link_fhitatten = max(best_link_fhitatten, fhits / total);

		std::cout << "Raw.BestMEANS = " << best_link_mean << endl;
		std::cout << "Raw.BestHITS = " << best_link_hitatten << endl;
		logging.record() << "Raw.BestMEANS = " << best_link_mean;
		logging.record() << "Raw.BestHITS = " << best_link_hitatten;
		std::cout << "Filter.BestMEANS = " << best_link_fmean << endl;
		std::cout << "Filter.BestHITS = " << best_link_fhitatten << endl;
		logging.record() << "Filter.BestMEANS = " << best_link_fmean;
		logging.record() << "Filter.BestHITS = " << best_link_fhitatten;
	}

	virtual void draw(const string& filename, const int radius, const int id_relation) const
	{
		return;
	}

	virtual void draw(const string& filename, const int radius,
		const int id_head, const int id_relation)
	{
		return;
	}

	virtual void report(const string& filename) const
	{
		return;
	}
public:
	~Model()
	{
		logging.record();
		if (be_deleted_data_model)
		{
			delete &data_model;
			delete &logging;
		}
	}

public:
	int count_entity() const
	{
		return data_model.set_entity.size();
	}

	int count_relation() const
	{
		return data_model.set_relation.size();
	}

	const DataModel& get_data_model() const
	{
		return data_model;
	}

public:
	virtual void save(const string& filename)
	{
		cout << "BAD";
	}

	virtual void load(const string& filename)
	{
		cout << "BAD";
	}

	virtual vec entity_representation(int entity_id) const
	{
		//cout << "BAD";
        vec ret = zeros<vec>(1);
        return ret;
        
	}

	virtual vec relation_representation(int relation_id) const
	{
        vec ret = zeros<vec>(1);
        return ret;
	}
};
