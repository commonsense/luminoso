map_value_inner = function (attr) {
    var val = this['attr'];
    if (this.words.length >= 2 && val) {
        this.words.forEach(function (word) {
            emit(word, {total: Math.exp(val)});
        })
    }
};
// MongoDB loses the closure when it sends this function to the server.
// We'll fill in the variable later with our own scope object, so for now
// it can be anything.
map_value = map_value_inner('dummy');

weighted_reduce_total = function(weight) {
    return function (key, values) {
        var tot = 0;
        for (var i=0; i<values.length; i++) {
            tot += weight * values[i].total;
        }
        return {total: tot};
    };
};
reduce_total = weighted_reduce_total(1);

collect_term_totals = function() {
    db.runCommand({mapreduce: 'relations',
                   map: map_value,
                   reduce: reduce_total,
                   scope: {weight: 1, attr: 'interestingness'},
                   out: 'term_totals'});
}

mapreduce_term = function(term, weight, collection) {
    db.runCommand({mapreduce: 'relations',
                   query: {words: term},
                   map: map_interestingness,
                   reduce: reduce_total,
                   scope: {weight: weight},
                   out: {reduce: collection}});
};

eigenvector_iteration = function(input_collection, output_collection, eigs) {
    // run MapReduce on every element in the current vector
    input_collection.find().forEach(function (entry) {
        mapreduce_term(entry['_id'], entry['total'], output_collection)
    }

    // Subtract out all previous eigenvectors
    var totals = [];
    for (var e=0; e<eigs.length; e++) {
        totals.push(0);
        var eig = eigs[e];
        output_collection.find().forEach(function(entry) {
            var prod = entry['total'] * (eig[entry['_id']] || 0);
            totals[e] += prod;
        }
    }

    // Divide by the Euclidean norm
    var mag_sq = 0.0;
    output_collection.find().forEach(function(entry) {
        var val = entry['total'];
        for (var e=0; e<eigs.length; e++) {
            val -= (eigs[e][entry['_id']] || 0) * totals[e];
        }
        mag_sq += val*val;
    }

    // Re-calculate the updated values and store them in the output collection
    output_collection.find().forEach(function(entry) {
        var val = entry['total'];
        for (var e=0; e<eigs.length; e++) {
            val -= (eigs[e][entry['_id']] || 0) * totals[e];
        }
        val /= Math.sqrt(mag_sq);
        output_collection.update({'_id': entry['_id']},
                                 {'$set': {total: val}});
    }
}
