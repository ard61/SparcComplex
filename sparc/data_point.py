import numpy as np
import json

from sparc.sparc_params import SparcParams

class DataPoint:
    """
    A set of data points of a SPARC simulation.

    Properties:
        sparc_params: SparcParams
    Simulation properties:
        num_trials = #(simulations)
        SER = array(section error rates)
        BER = array(bit error rates)
        avg_SER = average section error rate
        avg_BER = average bit error rate
        var_SER = variance of section error rate
        var_BER = variance of bit error rate
    """
    def __init__(self, sparc_params, num_trials, CERs, SERs, BERs):
        assert(CERs.size == num_trials)
        assert(SERs.size == num_trials)
        assert(BERs.size == num_trials)

        self.sparc_params = sparc_params
        self.num_trials = num_trials

        self.CERs = CERs
        self.SERs = SERs
        self.BERs = BERs

        self.compute_stats()

    def compute_stats(self):
        if self.num_trials == 0:
            self.avg_CER = 0
            self.avg_SER = 0
            self.avg_BER = 0

            self.stddev_CER = 0
            self.stddev_SER = 0
            self.stddev_BER = 0

        else:
            self.avg_CER = np.sum(self.CERs) / self.num_trials
            self.avg_SER = np.sum(self.SERs) / self.num_trials
            self.avg_BER = np.sum(self.BERs) / self.num_trials

            self.stddev_CER = np.sqrt(np.sum(self.CERs**2) / self.num_trials - self.avg_CER**2)
            self.stddev_SER = np.sqrt(np.sum(self.SERs**2) / self.num_trials - self.avg_SER**2)
            self.stddev_BER = np.sqrt(np.sum(self.BERs**2) / self.num_trials - self.avg_BER**2)

    def __str__(self):
        return '''Statistics for  {}   ; {} runs:
Codeword error rate: avg {}  ;  std-dev {}
Section  error rate: avg {}  ;  std-dev {}
Bit      error rate: avg {}  ;  std-dev {}'''.format(self.sparc_params,
                                                     self.num_trials,
                                                     self.avg_CER, self.stddev_CER,
                                                     self.avg_SER, self.stddev_SER,
                                                     self.avg_BER, self.stddev_BER)


    @classmethod
    def combine(cls, data_points):
        sparc_params = data_points[0].sparc_params
        for data_point in data_points:
            assert(data_point.sparc_params == sparc_params)
        
        num_trials = sum((data_point.num_trials for data_point in data_points))
        
        CERs = np.concatenate([data_point.CERs for data_point in data_points])
        SERs = np.concatenate([data_point.SERs for data_point in data_points])
        BERs = np.concatenate([data_point.BERs for data_point in data_points])
        
        return cls(sparc_params, num_trials, CERs, SERs, BERs)

    def to_json(self):
        return json.dumps({
            'sparc_params': self.sparc_params.to_json(),
            'num_trials': int(self.num_trials),
            'CERs': self.CERs.tolist(),
            'SERs': self.SERs.tolist(),
            'BERs': self.BERs.tolist()
        })

    @classmethod
    def none(cls, sparc_params):
        return cls(sparc_params, 0, np.array([]), np.array([]), np.array([]))

    @classmethod
    def from_json(cls, string):
        p = json.loads(string)
        return cls(SparcParams.from_json(p['sparc_params']), p['num_trials'], np.array(p['CERs']), np.array(p['SERs']), np.array(p['BERs']))
