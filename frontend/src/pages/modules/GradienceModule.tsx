import React, { useState } from 'react';
import { TrendingUp, Plus, Trash2, ArrowRight } from 'lucide-react';
import ModuleHeader from '../../components/ModuleHeader';
import gradienceIcon from '../../assets/gradience.png';
import { gradienceApi } from '../../api/client';
import { motion } from 'framer-motion';

interface TableRow {
  name: string;
  value: string;
}

const DynamicTable = ({
  headers,
  rows,
  onChange,
  onAdd,
  onRemove,
  namePlaceholder = "e.g. C(=O)O",
  valuePlaceholder = "0.0265"
}: {
  headers: string[],
  rows: any[],
  onChange: (index: number, field: string, value: string) => void,
  onAdd: () => void,
  onRemove: (index: number) => void,
  type?: 'solvent',
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
                type="text"
                value={row.name}
                onChange={(e) => onChange(idx, 'name', e.target.value)}
                className="w-full px-3 py-2 rounded-lg border-gray-300 border shadow-sm focus:border-purple-500 focus:ring-purple-500 focus:outline-none transition-all text-sm"
                placeholder={namePlaceholder}
              />
            </td>
            <td className="p-2">
              <input
                type="number"
                value={row.value}
                onChange={(e) => onChange(idx, 'value', e.target.value)}
                className="w-full px-3 py-2 rounded-lg border-gray-300 border shadow-sm focus:border-purple-500 focus:ring-purple-500 focus:outline-none transition-all text-sm"
                placeholder={valuePlaceholder}
              />
            </td>
            <td className="p-2 pr-4 text-center">
              <button
                type="button"
                onClick={() => onRemove(idx)}
                className="p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-all opacity-0 group-hover:opacity-100"
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

const GradienceModule: React.FC = () => {
  const [reactants, setReactants] = useState<string[]>(['']);
  const [solvent, setSolvent] = useState('');

  // method parameters
  const [solventASolvents, setSolventASolvents] = useState<TableRow[]>([{ name: 'O', value: '95.0' }, { name: 'CO', value: '5.0' }]);
  const [solventAAdditives, setSolventAAdditives] = useState<TableRow[]>([]);

  const [solventBSolvents, setSolventBSolvents] = useState<TableRow[]>([{ name: 'CC#N', value: '100.0' }]);
  const [solventBAdditives, setSolventBAdditives] = useState<TableRow[]>([]);

  const [colType, setColType] = useState('RP');
  const [colDiameter, setColDiameter] = useState('4.6');
  const [colLength, setColLength] = useState('150');
  const [particleSize, setParticleSize] = useState('5');

  const [flowRate, setFlowRate] = useState('1.0');
  const [temp, setTemp] = useState('40.0');
  const [methodLength, setMethodLength] = useState('15.0');

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any | null>(null);

  // handlers
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
  const handleAddRow = (setter: React.Dispatch<React.SetStateAction<any[]>>) => {
    setter(prev => [...prev, { name: '', value: '' }]);
  };
  const handleRemoveRow = (setter: React.Dispatch<React.SetStateAction<any[]>>, idx: number) => {
    setter(prev => prev.filter((_, i) => i !== idx));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    // method parameters JSON
    const formatSolventDict = (rows: TableRow[]) => rows.reduce((acc, r) => {
      if (r.name && r.value) acc[r.name] = parseFloat(r.value);
      return acc;
    }, {} as Record<string, number>);

    const lcmsConfig = {
      solvents: {
        'A': [formatSolventDict(solventASolvents), formatSolventDict(solventAAdditives)],
        'B': [formatSolventDict(solventBSolvents), formatSolventDict(solventBAdditives)]
      },
      column: [colType, parseFloat(colDiameter), parseFloat(colLength), parseFloat(particleSize)],
      flow_rate: parseFloat(flowRate),
      temp: parseFloat(temp),
      method_length: parseFloat(methodLength)
    };

    try {
      const validReactants = reactants.filter(r => r.trim() !== '');
      const response = await gradienceApi.optimize({
        reactant_smiles: validReactants,
        solvent_smiles: solvent,
        lcms_config: lcmsConfig
      });
      setResult(response.data);
    } catch (error) {
      console.error(error);
      alert('Error optimizing gradient');
    } finally {
      setLoading(false);
    }
  };

  // gradient chart
  const GradientChart = ({ gradient }: { gradient: [number, number][] }) => {
    if (!gradient || gradient.length === 0) return null;

    const maxX = Math.max(...gradient.map(p => p[0]));
    const padding = 60;
    const width = 600;
    const height = 300;

    const xScale = (x: number) => (x / maxX) * (width - 2 * padding) + padding;
    const yScale = (y: number) => height - padding - (y / 100) * (height - 2 * padding);

    const pointsString = gradient.map(p => `${xScale(p[0])},${yScale(p[1])}`).join(' ');

    return (
      <div className="w-full overflow-x-auto flex justify-center">
        <svg width="100%" height="100%" viewBox={`0 0 ${width} ${height}`} className="min-w-[500px] max-w-[600px]">
          {/* Grid Lines */}
          {[0, 25, 50, 75, 100].map(y => (
            <g key={y}>
              <line
                x1={padding}
                y1={yScale(y)}
                x2={width - padding}
                y2={yScale(y)}
                stroke="#e5e7eb"
                strokeDasharray="4 4"
              />
              <text x={padding - 10} y={yScale(y) + 4} textAnchor="end" fontSize="10" fill="#9ca3af">{y}%</text>
            </g>
          ))}

          {/* Axes */}
          <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#4b5563" strokeWidth="2" />
          <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#4b5563" strokeWidth="2" />

          {/* Line */}
          <polyline
            points={pointsString}
            fill="none"
            stroke="#9333ea"
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {/* Points */}
          {gradient.map((p, i) => (
            <circle
              key={i}
              cx={xScale(p[0])}
              cy={yScale(p[1])}
              r="4"
              fill="#9333ea"
              className="hover:r-6 transition-all"
            >
              <title>{`${p[0].toFixed(2)} min, ${p[1].toFixed(1)}% B`}</title>
            </circle>
          ))}

          {/* Labels */}
          <text x={width / 2} y={height - 10} textAnchor="middle" fontSize="12" fill="#4b5563" fontWeight="bold">Time (min)</text>
          <text x={25} y={height / 2} textAnchor="middle" transform={`rotate(-90, 25, ${height / 2})`} fontSize="12" fill="#4b5563" fontWeight="bold">% Solvent B</text>
        </svg>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <ModuleHeader
        title="Gradience"
        description={
          <>
            Automated LC-MS gradient optimization using Trust-Based Bayesian Optimization (TuRBO) to maximize peak separation.
            <br /><br />Official documentation for Gradience can be found{" "}
            <a
              href="https://github.com/natelgrw/gradience"
              target="blank"
              rel="noopener noreferrer"
              className="text-white underline"
            >
              here
            </a>.
          </>
        }
        icon={gradienceIcon}
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
            <form onSubmit={handleSubmit} className="space-y-8">

              {/* Reaction Details */}
              <div>
                <h2 className="text-2xl font-bold text-gray-900 mb-6">Reaction Details</h2>
                <div className="space-y-4">
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
                        onAdd={() => handleAddRow(setSolventASolvents)}
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
                        onAdd={() => handleAddRow(setSolventAAdditives)}
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
                        onAdd={() => handleAddRow(setSolventBSolvents)}
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
                        onAdd={() => handleAddRow(setSolventBAdditives)}
                        onRemove={(i) => handleRemoveRow(setSolventBAdditives, i)}
                        namePlaceholder="e.g. C(=O)O"
                        valuePlaceholder="e.g. 0.0265"
                      />
                    </div>
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
                  </div>
                </div>

                {/* Conditions */}
                <div className="mb-6">
                  <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                    Conditions
                  </h3>
                  <div className="pl-0 grid grid-cols-3 gap-4">
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">Flow Rate (mL/min)</label>
                      <input
                        type="number"
                        step="0.1"
                        value={flowRate}
                        onChange={e => setFlowRate(e.target.value)}
                        className="w-full px-3 py-2 rounded-lg border-gray-300 border shadow-sm focus:border-purple-500 focus:ring-red-500 focus:outline-none transition-all text-sm"
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">Temperature (°C)</label>
                      <input
                        type="number"
                        step="0.1"
                        value={temp}
                        onChange={e => setTemp(e.target.value)}
                        className="w-full px-3 py-2 rounded-lg border-gray-300 border shadow-sm focus:border-purple-500 focus:ring-red-500 focus:outline-none transition-all text-sm"
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">Method Length (min)</label>
                      <input
                        type="number"
                        step="1.0"
                        value={methodLength}
                        onChange={e => setMethodLength(e.target.value)}
                        className="w-full px-3 py-2 rounded-lg border-gray-300 border shadow-sm focus:border-purple-500 focus:ring-red-500 focus:outline-none transition-all text-sm"
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
                    Optimizing...
                  </span>
                ) : (
                  <>
                    Optimize Gradient <ArrowRight className="ml-2 w-5 h-5" />
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
              <h2 className="text-2xl font-bold text-gray-900 mb-6">Optimized Gradient</h2>

              {result ? (
                <div>
                  <div className="mb-6 p-4 bg-purple-50 rounded-xl border border-purple-100 flex justify-between items-center">
                    <div>
                      <div className="text-sm text-purple-600 font-medium">Separation Score</div>
                      <div className="text-3xl font-bold text-gray-900">{result.score.toFixed(4)}</div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm text-purple-600 font-medium">Method Length</div>
                      <div className="text-xl font-bold text-gray-900">{result.gradient[result.gradient.length - 1][0]} min</div>
                    </div>
                  </div>

                  <div className="mb-8">
                    <h3 className="text-lg font-semibold mb-4">Gradient Profile</h3>
                    <GradientChart gradient={result.gradient} />
                  </div>

                  <div>
                    <h3 className="text-lg font-semibold mb-3">Timetable</h3>
                    <div className="bg-gray-50 rounded-xl overflow-hidden border border-gray-200">
                      <table className="min-w-full divide-y divide-gray-200">
                        <thead className="bg-gray-100">
                          <tr>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Time (min)</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">% Solvent B</th>
                          </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-gray-200">
                          {result.gradient.map((point: any, i: number) => (
                            <tr key={i}>
                              <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{point[0].toFixed(2)}</td>
                              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{point[1].toFixed(1)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex-1 flex flex-col items-center justify-center text-gray-400">
                  <TrendingUp className="w-16 h-16 mb-4 opacity-20" />
                  <p>Enter reaction details to optimize gradient</p>
                </div>
              )}
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default GradienceModule;
