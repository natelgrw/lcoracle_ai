import React, { useState, useRef } from 'react';
import { Clock, ArrowRight, Upload, Download, Plus, Trash2 } from 'lucide-react';
import ModuleHeader from '../../components/ModuleHeader';
import retinaIcon from '../../assets/retina.png';
import { retinaApi } from '../../api/client';
import { motion } from 'framer-motion';

interface TableRow {
  name: string;
  value: string;
}

interface GradientRow {
  time: string;
  pctB: string;
}

interface BatchResult {
  compound: string;
  retention_time: number;
}

const DynamicTable = ({
  headers,
  rows,
  onChange,
  onAdd,
  onRemove,
  type = 'solvent',
  namePlaceholder = "e.g. C(=O)O",
  valuePlaceholder = "0.0265"
}: {
  headers: string[],
  rows: any[],
  onChange: (index: number, field: string, value: string) => void,
  onAdd: () => void,
  onRemove: (index: number) => void,
  type?: 'solvent' | 'gradient',
  namePlaceholder?: string,
  valuePlaceholder?: string
}) => (
  <div className="border border-gray-200 rounded-xl overflow-hidden mb-4 shadow-sm">
    <table className="w-full text-sm">
      <thead className="bg-gray-50 border-b border-gray-100">
        <tr>
          {headers.map((h, i) => (
            <th key={i} className="px-4 py-3 text-left font-semibold text-gray-500 uppercase tracking-wider text-xs">{h}</th>
          ))}
          <th className="w-10"></th>
        </tr>
      </thead>
      <tbody className="divide-y divide-gray-100 bg-white">
        {rows.map((row, idx) => (
          <tr key={idx} className="group hover:bg-gray-50 transition-colors">
            <td className="p-2 pl-4">
              <input
                type={type === 'gradient' ? "number" : "text"}
                value={type === 'solvent' ? row.name : row.time}
                onChange={(e) => onChange(idx, type === 'solvent' ? 'name' : 'time', e.target.value)}
                className="w-full px-3 py-2 rounded-lg border-gray-300 border shadow-sm focus:border-purple-500 focus:ring-purple-500 focus:outline-none transition-all text-sm"
                placeholder={type === 'solvent' ? namePlaceholder : "0.0265"}
              />
            </td>
            <td className="p-2">
              <input
                type="number"
                value={type === 'solvent' ? row.value : row.pctB}
                onChange={(e) => onChange(idx, type === 'solvent' ? 'value' : 'pctB', e.target.value)}
                className="w-full px-3 py-2 rounded-lg border-gray-300 border shadow-sm focus:border-purple-500 focus:ring-purple-500 focus:outline-none transition-all text-sm"
                placeholder={type === 'solvent' ? valuePlaceholder : "5"}
              />
            </td>
            <td className="p-2 pr-4 text-center">
              <button
                type="button"
                onClick={() => onRemove(idx)}
                className="p-2 rounded-lg transition-all text-red-500 opacity-100 md:text-gray-400 md:opacity-0 md:group-hover:opacity-100 md:hover:text-red-500 md:hover:bg-red-50"
              >
                <Trash2 size={16} />
              </button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
    <button
      type="button"
      onClick={onAdd}
      className="w-full py-3 bg-gray-50 text-purple-600 text-sm font-medium hover:bg-gray-100 hover:text-purple-700 transition-colors flex items-center justify-center gap-2 border-t border-gray-100"
    >
      <Plus size={16} /> Add Row
    </button>
  </div>
);

const RetinaModule: React.FC = () => {
  const [compound, setCompound] = useState('CCO');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [solventASolvents, setSolventASolvents] = useState<TableRow[]>([{ name: 'O', value: '95.0' }, { name: 'CO', value: '5.0' }]);
  const [solventAAdditives, setSolventAAdditives] = useState<TableRow[]>([]);

  const [solventBSolvents, setSolventBSolvents] = useState<TableRow[]>([{ name: 'CC#N', value: '100.0' }]);
  const [solventBAdditives, setSolventBAdditives] = useState<TableRow[]>([]);

  const [gradient, setGradient] = useState<GradientRow[]>([
    { time: '0', pctB: '5' },
    { time: '10', pctB: '95' },
    { time: '15', pctB: '95' }
  ]);

  const [colType, setColType] = useState('RP');
  const [colDiameter, setColDiameter] = useState('4.6');
  const [colLength, setColLength] = useState('150');
  const [particleSize, setParticleSize] = useState('5');

  const [flowRate, setFlowRate] = useState('1.0');
  const [temp, setTemp] = useState('40.0');

  // Processing & Results
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<number | null>(null);
  const [batchResults, setBatchResults] = useState<BatchResult[]>([]);
  const [batchCompounds, setBatchCompounds] = useState<string[]>([]); // To store loaded CSV compounds

  // --- Handlers ---

  // Table Handlers
  const handleTableChange = (
    setter: React.Dispatch<React.SetStateAction<any[]>>,
    idx: number,
    field: string,
    val: string
  ) => {
    setter(prev => {
      const newRows = [...prev];
      newRows[idx] = { ...newRows[idx], [field]: val };
      return newRows;
    });
  };

  const handleAddRow = (setter: React.Dispatch<React.SetStateAction<any[]>>, type: 'solvent' | 'gradient') => {
    setter(prev => [...prev, type === 'solvent' ? { name: '', value: '' } : { time: '', pctB: '' }]);
  };

  const handleRemoveRow = (setter: React.Dispatch<React.SetStateAction<any[]>>, idx: number) => {
    setter(prev => prev.filter((_, i) => i !== idx));
  };

  // File Upload
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setResult(null);
    setBatchResults([]);

    try {
      const text = await file.text();
      const lines = text.split('\n').map(l => l.trim()).filter(l => l);
      // CSV no header, col 1 is SMILES
      const compounds = lines.map(line => line.split(',')[0].trim()).filter(s => s);
      setBatchCompounds(compounds);
      // Reset single compound input to indicate batch mode or just leave it?
      // User said "repetitive iterations". We will prioritize batchCompounds if present when predicting.
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

  // Prediction Logic
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    // Prepare data structures
    const formatSolventDict = (rows: TableRow[]) => rows.reduce((acc, r) => {
      if (r.name && r.value) acc[r.name] = parseFloat(r.value);
      return acc;
    }, {} as Record<string, number>);

    const solvents = {
      'A': [formatSolventDict(solventASolvents), formatSolventDict(solventAAdditives)],
      'B': [formatSolventDict(solventBSolvents), formatSolventDict(solventBAdditives)]
    };

    const gradientData = gradient.map(r => [parseFloat(r.time), parseFloat(r.pctB)]);
    const columnData = [colType, parseFloat(colDiameter), parseFloat(colLength), parseFloat(particleSize)];

    const commonPayload = {
      solvents,
      gradient: gradientData,
      column: columnData,
      flow_rate: parseFloat(flowRate),
      temp: parseFloat(temp)
    };

    try {
      if (batchCompounds.length > 0) {
        // Batch Mode
        setBatchResults([]);
        const results: BatchResult[] = [];
        for (const cmp of batchCompounds) {
          try {
            const res = await retinaApi.predict({
              ...commonPayload,
              compound_smiles: cmp
            });
            results.push({ compound: cmp, retention_time: res.data.retention_time });
          } catch (err) {
            console.error(`Failed for ${cmp}`, err);
            results.push({ compound: cmp, retention_time: -1 }); // Error flag
          }
        }
        setBatchResults(results);
      } else {
        // Single Mode
        const response = await retinaApi.predict({
          ...commonPayload,
          compound_smiles: compound
        });
        setResult(response.data.retention_time);
      }
    } catch (error) {
      console.error(error);
      alert('Error during prediction');
    } finally {
      setLoading(false);
    }
  };

  const downloadCSV = () => {
    if (batchResults.length === 0) return;
    const header = "compound,retention_time_sec\n";
    const rows = batchResults.map(r => `${r.compound},${r.retention_time}`).join("\n");
    const blob = new Blob([header + rows], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'retina_predictions.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <ModuleHeader
        title="ReTiNA"
        description={
          <>
            Predict LC-MS retention times with high accuracy using the ReTiNA_XGB1 prediction model.
            <br /><br />Official documentation for ReTiNA models can be found{" "}
            <a
              href="https://github.com/natelgrw/retina_models"
              target="blank"
              rel="noopener noreferrer"
              className="text-white underline"
            >
              here
            </a>.
          </>
        }
        icon={retinaIcon}
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
            <form onSubmit={handleSubmit} className="space-y-8">
              {/* Compound Section */}
              <div>
                <h2 className="text-2xl font-bold text-gray-900 mb-6">Compound Details</h2>
                <div className="space-y-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Compound (SMILES)</label>
                    <input
                      type="text"
                      value={compound}
                      onChange={(e) => setCompound(e.target.value)}
                      className="w-full rounded-lg border-gray-300 shadow-sm focus:border-purple-500 focus:ring-purple-500 border p-2"
                      placeholder="e.g. CCO"
                      disabled={batchCompounds.length > 0}
                    />
                  </div>

                  {/* File Upload & Batch Indicator */}
                  <div className="relative py-4">
                    <div className="absolute inset-0 flex items-center">
                      <div className="w-full border-t border-gray-200"></div>
                    </div>
                    <div className="relative flex justify-center text-sm">
                      <span className="px-2 bg-white text-gray-500">Or upload batch (.csv)</span>
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
                      <label className="flex flex-col items-center justify-center w-full min-h-[6rem] h-auto p-4 text-center border-2 border-gray-300 border-dashed rounded-xl cursor-pointer bg-gray-50 hover:bg-gray-100 transition-colors">
                        <div className="flex flex-col items-center justify-center pt-5 pb-6">
                          <Upload className="w-6 h-6 text-gray-400 mb-2" />
                          <p className="mb-1 text-sm text-gray-500"><span className="font-semibold">Click to upload CSV</span></p>
                          <p className="text-xs text-gray-400">No header, compound SMILES in col 1</p>
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
              </div>

              {/* Method Parameters Title */}
              <div className="border-t border-gray-100 pt-8">
                <h2 className="text-2xl font-bold text-gray-900 mb-6">Method Parameters</h2>

                {/* Solvent Front A */}
                <div className="mb-8">
                  <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                    Solvent Front A
                  </h3>
                  <div className="space-y-6 pl-0">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">Solvents</label>
                      <DynamicTable
                        headers={['Solvent', 'Concentration (%)']}
                        rows={solventASolvents}
                        onChange={(i, f, v) => handleTableChange(setSolventASolvents, i, f, v)}
                        onAdd={() => handleAddRow(setSolventASolvents, 'solvent')}
                        onRemove={(i) => handleRemoveRow(setSolventASolvents, i)}
                        namePlaceholder="e.g. O"
                        valuePlaceholder="e.g. 95"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">Additives</label>
                      <DynamicTable
                        headers={['Additive', 'Concentration (M)']}
                        rows={solventAAdditives}
                        onChange={(i, f, v) => handleTableChange(setSolventAAdditives, i, f, v)}
                        onAdd={() => handleAddRow(setSolventAAdditives, 'solvent')}
                        onRemove={(i) => handleRemoveRow(setSolventAAdditives, i)}
                        namePlaceholder="e.g. C(=O)O"
                        valuePlaceholder="e.g. 0.0265"
                      />
                    </div>
                  </div>
                </div>

                {/* Solvent Front B */}
                <div className="mb-8">
                  <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                    Solvent Front B
                  </h3>
                  <div className="space-y-6 pl-0">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">Solvents</label>
                      <DynamicTable
                        headers={['Solvent', 'Concentration (%)']}
                        rows={solventBSolvents}
                        onChange={(i, f, v) => handleTableChange(setSolventBSolvents, i, f, v)}
                        onAdd={() => handleAddRow(setSolventBSolvents, 'solvent')}
                        onRemove={(i) => handleRemoveRow(setSolventBSolvents, i)}
                        namePlaceholder="e.g. O"
                        valuePlaceholder="e.g. 95"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">Additives</label>
                      <DynamicTable
                        headers={['Additive', 'Concentration (M)']}
                        rows={solventBAdditives}
                        onChange={(i, f, v) => handleTableChange(setSolventBAdditives, i, f, v)}
                        onAdd={() => handleAddRow(setSolventBAdditives, 'solvent')}
                        onRemove={(i) => handleRemoveRow(setSolventBAdditives, i)}
                        namePlaceholder="e.g. C(=O)O"
                        valuePlaceholder="e.g. 0.0265"
                      />
                    </div>
                  </div>
                </div>

                {/* Gradient */}
                <div className="mb-8">
                  <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                    Gradient Profile
                  </h3>
                  <div className="pl-0">
                    <DynamicTable
                      type="gradient"
                      headers={['Time (min)', '% Solvent Front B']}
                      rows={gradient}
                      onChange={(i, f, v) => handleTableChange(setGradient, i, f, v)}
                      onAdd={() => handleAddRow(setGradient, 'gradient')}
                      onRemove={(i) => handleRemoveRow(setGradient, i)}
                    />
                  </div>
                </div>

                {/* Column */}
                <div className="mb-8">
                  <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                    Column Configuration
                  </h3>
                  <div className="pl-0 space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="col-span-2">
                        <label className="block text-xs font-medium text-gray-700 mb-1">Column Type</label>
                        <div className="flex items-center p-1 bg-gray-100 rounded-xl w-fit">
                          <label className="cursor-pointer">
                            <input type="radio" name="colType" className="peer sr-only" checked={colType === 'RP'} onChange={() => setColType('RP')} />
                            <div className="px-6 py-2 rounded-lg text-sm font-medium text-gray-500 peer-checked:bg-white peer-checked:text-purple-600 peer-checked:shadow-sm transition-all">
                              Reverse Phase (RP)
                            </div>
                          </label>
                          <label className="cursor-pointer">
                            <input type="radio" name="colType" className="peer sr-only" checked={colType === 'HI'} onChange={() => setColType('HI')} />
                            <div className="px-6 py-2 rounded-lg text-sm font-medium text-gray-500 peer-checked:bg-white peer-checked:text-purple-600 peer-checked:shadow-sm transition-all">
                              HILIC (HI)
                            </div>
                          </label>
                        </div>
                      </div>
                    </div>
                    <div className="grid grid-cols-3 gap-4">
                      <div>
                        <label className="block text-xs font-medium text-gray-700 mb-1">Inner Diameter (mm)</label>
                        <input
                          type="number"
                          step="0.1"
                          value={colDiameter}
                          onChange={e => setColDiameter(e.target.value)}
                          className="w-full px-3 py-2 rounded-lg border-gray-300 border shadow-sm focus:border-purple-500 focus:ring-purple-500 focus:outline-none transition-all text-sm"
                        />
                      </div>
                      <div>
                        <label className="block text-xs font-medium text-gray-700 mb-1">Column Length (mm)</label>
                        <input
                          type="number"
                          step="1"
                          value={colLength}
                          onChange={e => setColLength(e.target.value)}
                          className="w-full px-3 py-2 rounded-lg border-gray-300 border shadow-sm focus:border-purple-500 focus:ring-green-500 focus:outline-none transition-all text-sm"
                        />
                      </div>
                      <div>
                        <label className="block text-xs font-medium text-gray-700 mb-1">Particle Size (µm)</label>
                        <input
                          type="number"
                          step="0.1"
                          value={particleSize}
                          onChange={e => setParticleSize(e.target.value)}
                          className="w-full px-3 py-2 rounded-lg border-gray-300 border shadow-sm focus:border-purple-500 focus:ring-green-500 focus:outline-none transition-all text-sm"
                        />
                      </div>
                    </div>
                  </div>
                </div>

                {/* Conditions */}
                <div className="mb-6">
                  <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                    Conditions
                  </h3>
                  <div className="pl-0 grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">Flow Rate (mL/min)</label>
                      <input
                        type="number"
                        step="0.1"
                        value={flowRate}
                        onChange={e => setFlowRate(e.target.value)}
                        className="w-full px-3 py-2 rounded-lg border-gray-300 border shadow-sm focus:border-purple-500 focus:ring-green-500 focus:outline-none transition-all text-sm"
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">Temperature (°C)</label>
                      <input
                        type="number"
                        step="0.1"
                        value={temp}
                        onChange={e => setTemp(e.target.value)}
                        className="w-full px-3 py-2 rounded-lg border-gray-300 border shadow-sm focus:border-purple-500 focus:ring-green-500 focus:outline-none transition-all text-sm"
                      />
                    </div>
                  </div>
                </div>

              </div>

              {/* Submit Button */}
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
                    Predict Retention Time <ArrowRight className="ml-2 w-5 h-5" />
                  </>
                )}
              </button>
            </form>
          </motion.div>

          {/* Results Display */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100 flex flex-col min-h-[400px] h-fit"
          >
            {batchResults.length > 0 ? (
              // Batch Results View
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
                      </div>
                      <div className="text-right">
                        <span className="text-lg font-bold text-purple-600">{res.retention_time > 0 ? res.retention_time.toFixed(1) : "Error"}</span>
                        <span className="text-xs text-gray-500 ml-1">sec</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : result !== null ? (
              // Single Result View
              <div className="text-center h-full flex flex-col justify-start pt-16">
                <h3 className="text-2xl font-bold text-gray-900 mb-8">Retention Time Results</h3>
                <div className="relative w-full h-16 bg-gray-100 rounded-full mb-8 overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.min((result / 1200) * 100, 100)}%` }} // Assuming max 20 min approx for visuals
                    className="absolute top-0 left-0 h-full bg-gradient-to-r from-purple-400 to-indigo-500"
                  />
                  <div className="absolute inset-0 flex items-center justify-center z-10 font-mono font-bold text-gray-700">
                    {(result / 60).toFixed(2)} min
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-8 text-center">
                  <div className="p-4 bg-gray-50 rounded-xl">
                    <div className="text-3xl font-bold text-gray-900">{result.toFixed(1)}</div>
                    <div className="text-sm text-gray-500">Seconds</div>
                  </div>
                  <div className="p-4 bg-gray-50 rounded-xl">
                    <div className="text-3xl font-bold text-gray-900">{(result / 60).toFixed(2)}</div>
                    <div className="text-sm text-gray-500">Minutes</div>
                  </div>
                </div>
              </div>
            ) : (
              // Empty State
              <div className="text-center text-gray-400 flex flex-col items-center justify-start h-full pt-16">
                <Clock className="w-16 h-16 mb-4 opacity-20" />
                <p>Enter parameters or upload batch to predict RT</p>
              </div>
            )}
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default RetinaModule;
