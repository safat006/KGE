/*
 * File:   lrucache.hpp
 * Author: Alexander Ponomarev
 *
 * Created on June 20, 2013, 5:09 PM
 */

#ifndef _LRUCACHE_HPP_INCLUDED_
#define    _LRUCACHE_HPP_INCLUDED_

#include <unordered_map>
#include <list>
#include <cstddef>
#include <stdexcept>


namespace cache {
    
    template<typename key_t, typename value_t>
    class lru_cache {
    public:
        typedef typename std::pair<key_t, value_t> key_value_pair_t;
        typedef typename std::list<key_value_pair_t>::iterator list_iterator_t;
        
        lru_cache(size_t max_size) :
        _max_size(max_size) {
        }
        
        struct key_hash : public std::unary_function<key_t, std::size_t>
        {
            std::size_t operator()(const key_t& k) const
            {
                return std::get<0>(k) ^ std::get<1>(k) ^ std::get<2>(k);
            }
        };
        
        struct key_equal : public std::binary_function<key_t, key_t, bool>
        {
            bool operator()(const key_t& v0, const key_t& v1) const
            {
                return (
                        std::get<0>(v0) == std::get<0>(v1) &&
                        std::get<1>(v0) == std::get<1>(v1) &&
                        std::get<2>(v0) == std::get<2>(v1)
                        );
            }
        };
        
        
        void clear(){
            _cache_items_list.clear();
            _cache_items_map.clear();
        }
        
        
        void put(const key_t& key, const value_t& value) {
            auto it = _cache_items_map.find(key);
            _cache_items_list.push_front(key_value_pair_t(key, value));
            if (it != _cache_items_map.end()) {
                _cache_items_list.erase(it->second);
                _cache_items_map.erase(it);
            }
            _cache_items_map[key] = _cache_items_list.begin();
            
            if (_cache_items_map.size() > _max_size) {
                auto last = _cache_items_list.end();
                last--;
                _cache_items_map.erase(last->first);
                _cache_items_list.pop_back();
            }
        }
        
        const value_t& get(const key_t& key) {
            auto it = _cache_items_map.find(key);
            if (it == _cache_items_map.end()) {
                throw std::range_error("There is no such key in cache");
            }
            else {
                _cache_items_list.splice(_cache_items_list.begin(), _cache_items_list, it->second);
                return it->second->second;
            }
        }
        
        bool exists(const key_t& key) const {
            return _cache_items_map.find(key) != _cache_items_map.end();
        }
        
        size_t size() const {
            return _cache_items_map.size();
        }
        
    private:
        std::list<key_value_pair_t> _cache_items_list;
        std::unordered_map<key_t, list_iterator_t, key_hash, key_equal> _cache_items_map;
        size_t _max_size;
    };
    
} // namespace cache

#endif    /* _LRUCACHE_HPP_INCLUDED_ */
