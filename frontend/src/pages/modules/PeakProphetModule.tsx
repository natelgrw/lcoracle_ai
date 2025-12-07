import React, { useState } from 'react';
import { Layers, ArrowRight, Upload, FileText, Plus, Trash2, Download } from 'lucide-react';
import ModuleHeader from '../../components/ModuleHeader';
import peakProphetIcon from '../../assets/peakprophet.png';
import { peakProphetApi } from '../../api/client';
import { motion } from 'framer-motion';

interface TableRow {
  name: string;
  value: string;
}

interface GradientRow {
  time: string;
  pctB: string;
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

const PeakProphetModule: React.FC = () => {
  const [reactants, setReactants] = useState<string[]>(['']);
  const [solvent, setSolvent] = useState('');

  // file inputs
  const [msFile, setMsFile] = useState<File | null>(null);
  const [uvFile, setUvFile] = useState<File | null>(null);

  // method parameters
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

  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any | null>(null);


  // reactant handlers
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

  // table handlers
  const handleTableChange = (setter: React.Dispatch<React.SetStateAction<any[]>>, idx: number, field: string, val: string) => {
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

  // file handlers
  const handleMsFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) setMsFile(e.target.files[0]);
  };
  const handleUvFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) setUvFile(e.target.files[0]);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!msFile) {
      alert('LC-MS data file is required');
      return;
    }
    if (!uvFile) {
      alert('UV-Vis data file is required');
      return;
    }

    setLoading(true);

    // method parameters JSON
    const formatSolventDict = (rows: TableRow[]) => rows.reduce((acc, r) => {
      if (r.name && r.value) acc[r.name] = parseFloat(r.value);
      return acc;
    }, {} as Record<string, number>);

    const methodParams = {
      solvents: {
        'A': [formatSolventDict(solventASolvents), formatSolventDict(solventAAdditives)],
        'B': [formatSolventDict(solventBSolvents), formatSolventDict(solventBAdditives)]
      },
      gradient: gradient.map(r => [parseFloat(r.time), parseFloat(r.pctB)]),
      column: [colType, parseFloat(colDiameter), parseFloat(colLength), parseFloat(particleSize)],
      flow_rate: parseFloat(flowRate),
      temp: parseFloat(temp)
    };

    const formData = new FormData();
    formData.append('reactants', JSON.stringify(reactants.filter(r => r.trim())));
    formData.append('solvent', solvent);
    formData.append('method_params', JSON.stringify(methodParams));
    formData.append('ms_file', msFile);
    if (uvFile) formData.append('uv_file', uvFile);

    try {
      const response = await peakProphetApi.predict(formData);
      setResults(response.data);
    } catch (error) {
      console.error(error);
      alert('Error analyzing data');
    } finally {
      setLoading(false);
    }
  };

  const downloadJSON = () => {
    if (!results) return;
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'peak_prophet_summary.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <ModuleHeader
        title="PeakProphet"
        description={
          <>
            Automated compound-peak matching for LC-MS data utilizing reaction input, LC-MS method parameters, and several machine learning pipelines.
            <br /><br />Official documentation for PeakProphet can be found{" "}
            <a
              href="https://github.com/natelgrw/peak_prophet"
              target="blank"
              rel="noopener noreferrer"
              className="text-white underline"
            >
              here
            </a>.
          </>
        }
        icon={peakProphetIcon}
        color="text-purple-600"
        gradient="bg-gradient-to-r from-purple-600 to-indigo-600"
        iconOffset="md:-mt-16"
      />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-start">

          {/* Analysis Setup Card */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100"
          >
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Analysis Setup</h2>
            <form onSubmit={handleSubmit} className="space-y-8">

              {/* Reactants Section */}
              <div>
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Reaction Components</h3>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Reactants (SMILES)</label>
                    {reactants.map((reactant, index) => (
                      <div key={index} className="flex gap-2 mb-2">
                        <input
                          type="text"
                          value={reactant}
                          onChange={(e) => updateReactant(index, e.target.value)}
                          className="flex-1 rounded-lg border-gray-300 shadow-sm focus:border-purple-500 focus:ring-purple-500 border p-2 text-sm"
                          placeholder="e.g. CC(=O)Cl"
                          required
                        />
                        {reactants.length > 1 && (
                          <button
                            type="button"
                            onClick={() => removeReactant(index)}
                            className="p-2 text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        )}
                      </div>
                    ))}
                    <button
                      type="button"
                      onClick={addReactant}
                      className="mt-2 flex items-center text-xs text-purple-600 hover:text-purple-800 font-medium"
                    >
                      <Plus className="w-3 h-3 mr-1" /> Add Reactant
                    </button>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Solvent (SMILES)</label>
                    <input
                      type="text"
                      value={solvent}
                      onChange={(e) => setSolvent(e.target.value)}
                      className="w-full rounded-lg border-gray-300 shadow-sm focus:border-purple-500 focus:ring-purple-500 border p-2 text-sm"
                      placeholder="e.g. CCO"
                      required
                    />
                  </div>
                </div>
              </div>

              {/* Data Upload Section */}
              <div className="border-t border-gray-100 pt-6">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Experimental Data</h3>

                <div className="space-y-4">
                  {/* LC-MS Upload */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">LC-MS Raw Data (mzXML)</label>
                    <div className="flex items-center justify-center w-full">
                      <label className={`flex flex-col items-center justify-center w-full min-h-[6rem] h-auto p-4 text-center border-2 border-dashed rounded-xl cursor-pointer hover:bg-gray-50 transition-colors ${msFile ? 'border-purple-400 bg-purple-50' : 'border-gray-300 bg-white'}`}>
                        <div className="flex flex-col items-center justify-center pt-5 pb-6">
                          {msFile ? (
                            <>
                              <FileText className="w-6 h-6 text-purple-500 mb-1" />
                              <p className="text-sm text-purple-700 font-medium truncate max-w-[200px]">{msFile.name}</p>
                            </>
                          ) : (
                            <>
                              <Upload className="w-6 h-6 text-gray-400 mb-1" />
                              <p className="text-xs text-gray-500">Upload mzXML</p>
                            </>
                          )}
                        </div>
                        <input type="file" className="hidden" accept=".mzXML" onChange={handleMsFileChange} />
                      </label>
                    </div>
                  </div>

                  {/* UV-Vis Upload */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">UV-Vis Data (CSV)</label>
                    <div className="flex items-center justify-center w-full">
                      <label className={`flex flex-col items-center justify-center w-full min-h-[6rem] h-auto p-4 text-center border-2 border-dashed rounded-xl cursor-pointer hover:bg-gray-50 transition-colors ${uvFile ? 'border-purple-400 bg-purple-50' : 'border-gray-300 bg-white'}`}>
                        <div className="flex flex-col items-center justify-center pt-5 pb-6">
                          {uvFile ? (
                            <>
                              <FileText className="w-6 h-6 text-purple-500 mb-1" />
                              <p className="text-sm text-purple-700 font-medium truncate max-w-[200px]">{uvFile.name}</p>
                            </>
                          ) : (
                            <>
                              <Upload className="w-6 h-6 text-gray-400 mb-1" />
                              <p className="text-xs text-gray-500">Upload CSV</p>
                            </>
                          )}
                        </div>
                        <input type="file" className="hidden" accept=".csv" onChange={handleUvFileChange} />
                      </label>
                    </div>
                  </div>
                </div>
              </div>

              {/* Method Parameters (Collapsible or just sectioned) */}
              <div className="border-t border-gray-100 pt-6">
                <h3 className="text-lg font-bold text-gray-900 mb-4">LC-MS Method Parameters</h3>

                {/* Solvent Front A */}
                <div className="mb-6">
                  <h4 className="text-sm font-semibold text-gray-700 mb-2">Solvent Front A</h4>
                  <DynamicTable headers={['Solvent', 'Conc (%)']} rows={solventASolvents} onChange={(i, f, v) => handleTableChange(setSolventASolvents, i, f, v)} onAdd={() => handleAddRow(setSolventASolvents, 'solvent')} onRemove={(i) => handleRemoveRow(setSolventASolvents, i)} namePlaceholder="e.g. O" valuePlaceholder="e.g. 95" />
                  <div className="mt-2"></div>
                  <DynamicTable headers={['Additive', 'Conc (M)']} rows={solventAAdditives} onChange={(i, f, v) => handleTableChange(setSolventAAdditives, i, f, v)} onAdd={() => handleAddRow(setSolventAAdditives, 'solvent')} onRemove={(i) => handleRemoveRow(setSolventAAdditives, i)} namePlaceholder="e.g. C(=O)O" valuePlaceholder="e.g. 0.0265" />
                </div>

                {/* Solvent Front B */}
                <div className="mb-6">
                  <h4 className="text-sm font-semibold text-gray-700 mb-2">Solvent Front B</h4>
                  <DynamicTable headers={['Solvent', 'Conc (%)']} rows={solventBSolvents} onChange={(i, f, v) => handleTableChange(setSolventBSolvents, i, f, v)} onAdd={() => handleAddRow(setSolventBSolvents, 'solvent')} onRemove={(i) => handleRemoveRow(setSolventBSolvents, i)} namePlaceholder="e.g. O" valuePlaceholder="e.g. 95" />
                  <div className="mt-2"></div>
                  <DynamicTable headers={['Additive', 'Conc (M)']} rows={solventBAdditives} onChange={(i, f, v) => handleTableChange(setSolventBAdditives, i, f, v)} onAdd={() => handleAddRow(setSolventBAdditives, 'solvent')} onRemove={(i) => handleRemoveRow(setSolventBAdditives, i)} namePlaceholder="e.g. C(=O)O" valuePlaceholder="e.g. 0.0265" />
                </div>

                {/* Gradient */}
                <div className="mb-6">
                  <h4 className="text-sm font-semibold text-gray-700 mb-2">Gradient Profile</h4>
                  <DynamicTable type="gradient" headers={['Time', '% B']} rows={gradient} onChange={(i, f, v) => handleTableChange(setGradient, i, f, v)} onAdd={() => handleAddRow(setGradient, 'gradient')} onRemove={(i) => handleRemoveRow(setGradient, i)} />
                </div>

                {/* Column & Conditions */}
                <div className="space-y-6">
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
                        className="w-full px-3 py-2 rounded-lg border-gray-300 border shadow-sm focus:border-purple-500 focus:ring-purple-500 focus:outline-none transition-all text-sm"
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">Particle Size (µm)</label>
                      <input
                        type="number"
                        step="0.1"
                        value={particleSize}
                        onChange={e => setParticleSize(e.target.value)}
                        className="w-full px-3 py-2 rounded-lg border-gray-300 border shadow-sm focus:border-purple-500 focus:ring-purple-500 focus:outline-none transition-all text-sm"
                      />
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">Flow Rate (mL/min)</label>
                      <input
                        type="number"
                        step="0.1"
                        value={flowRate}
                        onChange={e => setFlowRate(e.target.value)}
                        className="w-full px-3 py-2 rounded-lg border-gray-300 border shadow-sm focus:border-purple-500 focus:ring-purple-500 focus:outline-none transition-all text-sm"
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">Temperature (°C)</label>
                      <input
                        type="number"
                        step="0.1"
                        value={temp}
                        onChange={e => setTemp(e.target.value)}
                        className="w-full px-3 py-2 rounded-lg border-gray-300 border shadow-sm focus:border-purple-500 focus:ring-purple-500 focus:outline-none transition-all text-sm"
                      />
                    </div>
                  </div>
                </div>
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
                    Analyzing...
                  </span>
                ) : (
                  <>
                    Run PeakProphet <ArrowRight className="ml-2 w-5 h-5" />
                  </>
                )}
              </button>
            </form>
          </motion.div>

          {/* Results Display */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100 min-h-[600px]"
          >
            {results ? (
              <div className="flex flex-col h-full">
                <div className="flex justify-between items-center mb-6">
                  <div>
                    <h2 className="text-2xl font-bold text-gray-900">Analysis Results</h2>
                    <p className="text-sm text-gray-500 mt-1">Found {results.peaks?.length || 0} peaks</p>
                  </div>
                  <button
                    onClick={downloadJSON}
                    className="p-2 bg-purple-100 text-purple-700 rounded-full hover:bg-purple-200 transition-colors shadow-sm"
                    title="Download JSON Summary"
                  >
                    <Download className="w-5 h-5" />
                  </button>
                </div>

                <div className="flex-1 overflow-y-auto max-h-[800px] pr-2 space-y-4">
                  {results.peaks?.map((peak: any, idx: number) => (
                    <div key={idx} className="border border-gray-200 rounded-xl p-5 hover:border-purple-300 transition-all bg-gray-50/50">
                      <div className="flex justify-between items-start mb-4">
                        <div className="flex items-center space-x-4">
                          <div className="bg-white border border-gray-200 rounded-lg p-2 text-center min-w-[70px] shadow-sm">
                            <div className="text-xs text-gray-500 font-medium">RT</div>
                            <div className="font-bold text-gray-900 text-lg">{(peak.apex / 60).toFixed(2)}m</div>
                          </div>
                          {peak.lmax && (
                            <div className="bg-white border border-yellow-200 rounded-lg p-2 text-center min-w-[70px] shadow-sm">
                              <div className="text-xs text-yellow-600 font-medium">λmax</div>
                              <div className="font-bold text-yellow-700 text-lg">{peak.lmax.toFixed(0)}nm</div>
                            </div>
                          )}
                          {peak.mz && (
                            <div className="bg-white border border-blue-200 rounded-lg p-2 text-center min-w-[70px] shadow-sm">
                              <div className="text-xs text-blue-600 font-medium">m/z</div>
                              <div className="font-bold text-blue-700 text-lg">{peak.mz.toFixed(1)}</div>
                            </div>
                          )}
                        </div>

                        <div className="flex flex-col items-end">
                          <div className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wide ${peak.optimal_compound ? 'bg-green-100 text-green-700' : 'bg-gray-200 text-gray-600'}`}>
                            {peak.optimal_compound ? 'Assigned' : 'Unassigned'}
                          </div>

                        </div>
                      </div>

                      <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
                        <div className="text-xs text-gray-400 uppercase tracking-wide font-bold mb-2">Optimal Assignment</div>
                        {peak.optimal_compound ? (
                          <div className="font-mono text-sm text-gray-900 break-all">{peak.optimal_compound}</div>
                        ) : (
                          <div className="text-sm text-gray-400 italic">No confident assignment</div>
                        )}
                      </div>

                      {peak.potential_compounds && Object.keys(peak.potential_compounds).length > 0 && (
                        <div className="mt-4 pt-3 border-t border-gray-200">
                          <div className="text-xs text-gray-500 font-medium mb-2">Top Candidates</div>
                          <div className="space-y-2">
                            {Object.entries(peak.potential_compounds)
                              .sort(([, a], [, b]) => (b as number) - (a as number))
                              .slice(0, 3)
                              .map(([smiles, score]: [string, any], i) => (
                                <div key={i} className="flex justify-between text-xs items-center bg-white p-2 rounded border border-gray-100">
                                  <span className="font-mono text-gray-600 truncate flex-1 mr-4" title={smiles}>{smiles}</span>
                                  <span className="text-purple-600 font-medium">{Number(score).toFixed(3)}</span>
                                </div>
                              ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-gray-400 pt-16">
                <Layers className="w-20 h-20 mb-6 opacity-20" />
                <h3 className="text-xl font-medium text-gray-600 mb-2">Ready to Prophesize</h3>
                <p className="max-w-md text-center text-gray-400">
                  Configure your reaction parameters and upload experimental data to identify peaks.
                </p>
              </div>
            )}
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default PeakProphetModule;
