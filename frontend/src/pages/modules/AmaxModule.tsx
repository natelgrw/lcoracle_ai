import React, { useState, useRef } from 'react';
import { Activity, ArrowRight, Upload, Download } from 'lucide-react';
import ModuleHeader from '../../components/ModuleHeader';
import amaxIcon from '../../assets/amax.png';
import { amaxApi } from '../../api/client';
import { motion } from 'framer-motion';

interface BatchResult {
  compound: string;
  solvent: string;
  lambda_max: number;
}

const AmaxModule: React.FC = () => {
  const [compound, setCompound] = useState('');
  const [solvent, setSolvent] = useState('CCO');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<number | null>(null);
  const [batchResults, setBatchResults] = useState<BatchResult[]>([]);
  const [batchCompounds, setBatchCompounds] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setBatchResults([]);

    try {
      if (batchCompounds.length > 0) {
        const results: BatchResult[] = [];
        for (const smile of batchCompounds) {
          try {
            const response = await amaxApi.predict(smile, solvent);
            results.push({
              compound: smile,
              solvent: solvent,
              lambda_max: response.data.lambda_max
            });
          } catch (err) {
            console.error(`Failed for ${smile}`, err);
            results.push({ compound: smile, solvent, lambda_max: 0 });
          }
        }
        setBatchResults(results);
      } else {
        const response = await amaxApi.predict(compound, solvent);
        setResult(response.data.lambda_max);
      }
    } catch (error) {
      console.error(error);
      alert('Error predicting lambda max');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setResult(null);
    setBatchResults([]);

    try {
      const text = await file.text();
      const lines = text.split('\n').map(l => l.trim()).filter(l => l);
      const compounds = lines.map(line => line.split(',')[0].trim()).filter(s => s);
      setBatchCompounds(compounds);
    } catch (error) {
      console.error(error);
      alert('Error reading file');
    } finally {
      setLoading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const clearBatch = () => {
    setBatchCompounds([]);
    setBatchResults([]);
  };

  const downloadCSV = () => {
    if (batchResults.length === 0) return;
    const header = "compound,solvent,lambda_max\n";
    const rows = batchResults.map(r => `${r.compound},${r.solvent},${r.lambda_max}`).join("\n");
    const blob = new Blob([header + rows], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'amax_predictions.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const spectrumGradient = "bg-[linear-gradient(to_right,#9333ea,#2563eb,#38bdf8,#22c55e,#facc15,#f97316,#dc2626)]";

  return (
    <div className="min-h-screen bg-gray-50">
      <ModuleHeader
        title="AMAX"
        description={
          <>
            Predict UV-Vis absorption maxima for a compound-solvent combination
            using the AMAX_XGB1 prediction model.
            <br /><br />Official documentation for AMAX models can be found{" "}
            <a
              href="https://github.com/natelgrw/amax_models"
              target="blank"
              rel="noopener noreferrer"
              className="text-white underline"
            >
              here
            </a>.
          </>
        }
        icon={amaxIcon}
        color="text-purple-600"
        gradient="bg-gradient-to-r from-purple-600 to-indigo-600"
        iconOffset="-mt-16"
      />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
          {/* Input Form */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100 h-fit"
          >
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Compound Details</h2>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Compound (SMILES)</label>
                <input
                  type="text"
                  value={compound}
                  onChange={(e) => setCompound(e.target.value)}
                  className="w-full rounded-lg border-gray-300 shadow-sm focus:border-purple-500 focus:ring-purple-500 border p-2"
                  placeholder="e.g. c1ccccc1"
                  required={batchCompounds.length === 0 && !loading}
                  disabled={batchCompounds.length > 0}
                />
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
                <p className="mt-1 text-xs text-gray-500">Common solvents: Water (O), Methanol (CO), Acetonitrile (CC#N)</p>
              </div>

              <div className="flex flex-col gap-3">
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
                      Predict λmax <ArrowRight className="ml-2 w-5 h-5" />
                    </>
                  )}
                </button>

                <div className="relative py-4">
                  <div className="absolute inset-0 flex items-center">
                    <div className="w-full border-t border-gray-200"></div>
                  </div>
                  <div className="relative flex justify-center text-sm">
                    <span className="px-2 bg-white text-gray-500">Or upload batch</span>
                  </div>
                </div>

                {batchCompounds.length > 0 ? (
                  <div className="flex items-center justify-between p-4 bg-purple-50 border border-purple-200 rounded-xl">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-purple-100 rounded-lg">
                        <Upload className="w-5 h-5 text-purple-700" />
                      </div>
                      <div>
                        <p className="font-medium text-purple-900">Batch Loaded</p>
                        <p className="text-xs text-purple-700">{batchCompounds.length} compounds ready</p>
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={clearBatch}
                      className="text-sm text-red-500 hover:text-red-700 font-medium"
                    >
                      Clear
                    </button>
                  </div>
                ) : (
                  <div className="flex items-center justify-center w-full">
                    <label className="flex flex-col items-center justify-center w-full h-24 border-2 border-gray-300 border-dashed rounded-xl cursor-pointer bg-gray-50 hover:bg-gray-100 transition-colors">
                      <div className="flex flex-col items-center justify-center pt-5 pb-6">
                        <Upload className="w-6 h-6 text-gray-400 mb-2" />
                        <p className="mb-1 text-sm text-gray-500"><span className="font-semibold">Click to upload CSV</span></p>
                        <p className="text-xs text-gray-400">No header, compound SMILES in col 1, solvent SMILES in col 2</p>
                      </div>
                      <input
                        ref={fileInputRef}
                        type="file"
                        className="hidden"
                        accept=".csv,.txt"
                        onChange={handleFileUpload}
                        disabled={loading}
                      />
                    </label>
                  </div>
                )}
              </div>
            </form>
          </motion.div>

          {/* Results Display */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100 flex flex-col min-h-[400px]"
          >
            {batchResults.length > 0 ? (
              <div className="flex flex-col h-full">
                <div className="flex justify-between items-center mb-6">
                  <h3 className="text-2xl font-bold text-gray-900">Batch Results</h3>
                  <button
                    onClick={downloadCSV}
                    className="p-2 bg-purple-100 text-purple-700 rounded-full hover:bg-purple-200 transition-colors shadow-sm"
                    title="Download CSV"
                  >
                    <Download className="w-5 h-5" />
                  </button>
                </div>
                <div className="flex-1 overflow-y-auto max-h-[600px] space-y-3">
                  {batchResults.map((res, idx) => (
                    <div key={idx} className="p-4 rounded-xl bg-gray-50 border border-gray-200 flex justify-between items-center">
                      <div className="flex-1 min-w-0 mr-4">
                        <p className="font-mono text-sm text-gray-900 truncate" title={res.compound}>{res.compound}</p>
                        <p className="text-xs text-gray-500">Solvent: {res.solvent}</p>
                      </div>
                      <div className="text-right">
                        <span className="text-lg font-bold text-purple-600">{res.lambda_max.toFixed(1)}</span>
                        <span className="text-xs text-gray-500 ml-1">nm</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : result !== null ? (
              <div className="text-center h-full flex flex-col justify-center">
                <motion.div
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  className="w-48 h-48 rounded-full bg-gradient-to-br from-purple-100 to-indigo-100 flex items-center justify-center mx-auto mb-6 shadow-inner relative overflow-hidden"
                >
                  <div className="z-10 relative">
                    <motion.span
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="text-5xl font-bold text-gray-900 block"
                    >
                      {result.toFixed(1)}
                    </motion.span>
                    <span className="text-xl text-gray-500 block mt-1">nm</span>
                  </div>
                  {/* Animated Background Ring */}
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
                    className="absolute inset-0 border-4 border-dashed border-purple-300/50 rounded-full"
                  />
                </motion.div>

                <h3 className="text-2xl font-bold text-gray-900 mb-2">Predicted Absorption Maximum</h3>
                <p className="text-gray-500 mb-8">
                  Compound in {solvent === 'O' ? 'Water' : solvent === 'CO' ? 'Methanol' : 'Solvent'}
                </p>

                {/* Visual Spectrum Representation */}
                <div className={`w-full h-12 rounded-full ${spectrumGradient} relative opacity-90 shadow-inner overflow-hidden`}>
                  <motion.div
                    initial={{ left: "0%" }}
                    animate={{ left: `${Math.min(Math.max((result / 800) * 100, 0), 100)}%` }}
                    transition={{ type: "spring", stiffness: 50, damping: 15 }}
                    className="absolute top-0 bottom-0 w-1 bg-white border-x border-black/20 shadow-[0_0_10px_rgba(0,0,0,0.5)] transform -translate-x-1/2 z-10"
                  >
                    <div className="absolute -top-1 left-1/2 transform -translate-x-1/2 -translate-y-full text-xs font-bold text-gray-700 bg-white px-1 rounded shadow-sm whitespace-nowrap">
                      {result.toFixed(0)} nm
                    </div>
                  </motion.div>
                </div>
                <div className="flex justify-between text-xs text-gray-400 mt-2 px-1 font-mono">
                  <span>0nm</span>
                  <span>400nm</span>
                  <span>800nm</span>
                </div>
              </div>
            ) : (
              // empty results state
              <div className="text-center text-gray-400 flex flex-col items-center justify-center h-full">
                <Activity className="w-16 h-16 mb-4 opacity-20" />
                <p>Enter compound or upload CSV to predict λmax</p>
              </div>
            )}
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default AmaxModule;
