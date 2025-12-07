import React, { useState } from 'react';
import { Search, Plus, Trash2, ArrowRight, Download } from 'lucide-react';
import ModuleHeader from '../../components/ModuleHeader';
import askcosIcon from '../../assets/askcos.png';
import { askcosApi } from '../../api/client';
import { motion } from 'framer-motion';

const AskcosModule: React.FC = () => {
  const [reactants, setReactants] = useState<string[]>(['']);
  const [solvent, setSolvent] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any[]>([]);

  const addReactant = () => setReactants([...reactants, '']);
  const removeReactant = (index: number) => {
    const newReactants = [...reactants];
    newReactants.splice(index, 1);
    setReactants(newReactants);
  };
  const updateReactant = (index: number, value: string) => {
    const newReactants = [...reactants];
    newReactants[index] = value;
    setReactants(newReactants);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const validReactants = reactants.filter(r => r.trim() !== '');
      const response = await askcosApi.predict(validReactants, solvent);
      setResults(response.data.products);
    } catch (error) {
      console.error(error);
      alert('Error predicting products');
    } finally {
      setLoading(false);
    }
  };

  const inputSet = new Set([
    ...reactants.map(r => r.trim()),
    solvent.trim()
  ].filter(s => s !== ''));

  const filteredResults = results.filter(product => {
    if (inputSet.has(product.smiles.trim())) return false;

    const productPart = product.smiles.split('>>').pop()?.trim();
    if (productPart && inputSet.has(productPart)) return false;

    return true;
  });

  const downloadCSV = () => {
    if (filteredResults.length === 0) return;

    const header = "smiles,probability,mol_weight\n";
    const rows = filteredResults.map(p => {
      const cleanSmiles = p.smiles.split('>>').pop() || p.smiles;
      return `${cleanSmiles},${p.probability},${p.mol_weight}`;
    }).join("\n");

    const blob = new Blob([header + rows], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'askcos_predictions.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <ModuleHeader
        title="ASKCOS"
        description={
          <>
            Predict reaction products from reactant SMILES using MIT ASKCOS software.
            <br /><br />Official documentation for ASKCOS can be found{" "}
            <a
              href="https://askcos-docs.mit.edu"
              target="blank"
              rel="noopener noreferrer"
              className="text-white underline"
            >
              here
            </a>.
          </>
        }
        icon={askcosIcon}
        color="text-purple-600"
        gradient="bg-gradient-to-r from-purple-600 to-indigo-600"
        iconOffset="md:-mt-16"
      />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
          {/* Input Form */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100"
          >
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Reaction Parameters</h2>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Reactants (SMILES)</label>
                {reactants.map((reactant, index) => (
                  <div key={index} className="flex gap-2 mb-2">
                    <input
                      type="text"
                      value={reactant}
                      onChange={(e) => updateReactant(index, e.target.value)}
                      className="flex-1 rounded-lg border-gray-300 shadow-sm focus:border-purple-500 focus:ring-purple-500 border p-2"
                      placeholder="e.g. CC(=O)Cl"
                      required
                    />
                    {reactants.length > 1 && (
                      <button
                        type="button"
                        onClick={() => removeReactant(index)}
                        className="p-2 text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                      >
                        <Trash2 className="w-5 h-5" />
                      </button>
                    )}
                  </div>
                ))}
                <button
                  type="button"
                  onClick={addReactant}
                  className="mt-2 flex items-center text-sm text-purple-600 hover:text-purple-800 font-medium"
                >
                  <Plus className="w-4 h-4 mr-1" /> Add Reactant
                </button>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Solvent (SMILES)</label>
                <input
                  type="text"
                  value={solvent}
                  onChange={(e) => setSolvent(e.target.value)}
                  className="w-full rounded-lg border-gray-300 shadow-sm focus:border-purple-500 focus:ring-purple-500 border p-2"
                  placeholder="e.g. CCO"
                  required
                />
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full py-3 px-4 bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-xl font-bold shadow-lg hover:shadow-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
              >
                {loading ? (
                  <span className="flex items-center">
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Processing...
                  </span>
                ) : (
                  <>
                    Run Prediction <ArrowRight className="ml-2 w-5 h-5" />
                  </>
                )}
              </button>
            </form>
          </motion.div>

          {/* Results Display */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
            <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100 min-h-[400px] flex flex-col">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-bold text-gray-900">Predicted Products</h2>
                {filteredResults.length > 0 && (
                  <button
                    onClick={downloadCSV}
                    className="p-2 bg-purple-100 text-purple-600 rounded-full hover:bg-purple-200 transition-colors shadow-sm"
                    title="Download CSV"
                  >
                    <Download className="w-5 h-5" />
                  </button>
                )}
              </div>

              {filteredResults.length > 0 ? (
                <div className="space-y-4 flex-1 overflow-y-auto max-h-[600px]">
                  {filteredResults.map((product, idx) => (
                    <div key={idx} className="p-4 rounded-xl bg-gray-50 border border-gray-200 hover:border-purple-300 transition-colors">
                      <div className="flex justify-between items-start">
                        <div>
                          <p className="font-mono text-sm text-purple-600 break-all">
                            {product.smiles.split('>>').pop()}
                          </p>
                          <div className="mt-2 flex space-x-4 text-sm text-gray-500">
                            <span>MW: {parseFloat(product.mol_weight).toFixed(2)}</span>
                            <span>Prob: {(parseFloat(product.probability) * 100).toFixed(1)}%</span>
                          </div>
                        </div>
                        <div className="bg-purple-100 text-purple-700 px-3 py-1 rounded-full text-xs font-bold">
                          Rank #{idx + 1}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex-1 flex flex-col items-center justify-center text-gray-400 text-center">
                  <Search className="w-16 h-16 mb-4 opacity-20" />
                  <p>Run a prediction to see results</p>
                </div>
              )}
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default AskcosModule;
