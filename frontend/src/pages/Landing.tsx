import React from 'react';
import { motion } from 'framer-motion';
import { ChevronRight } from 'lucide-react';
import { Link } from 'react-router-dom';
import PeripheralGraph from '../components/PeripheralGraph';
import ChromatogramAnimation from '../components/ChromatogramAnimation';
import InteractiveBackground from '../components/InteractiveBackground';
import askcosLogo from '../assets/askcos.png';
import amaxLogo from '../assets/amax.png';
import retinaLogo from '../assets/retina.png';
import peakProphetLogo from '../assets/peakprophet.png';
import gradienceLogo from '../assets/gradience.png';

const Landing: React.FC = () => {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { duration: 0.3, staggerChildren: 0.05 }
    }
  };

  const itemVariants = {
    hidden: { y: 10, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: { duration: 0.3 }
    }
  };

  const modules = [
    { name: 'ASKCOS', description: 'Automated product prediction from reactant input.', image: askcosLogo, color: 'text-blue-600', bg: 'bg-blue-50/50', path: '/askcos' },
    { name: 'AMAX', description: 'Accurate compound UV-Vis absorption maxima prediction.', image: amaxLogo, color: 'text-yellow-500', bg: 'bg-yellow-50/50', path: '/amax' },
    { name: 'ReTiNA', description: 'Precise compound LC-MS retention time prediction.', image: retinaLogo, color: 'text-green-600', bg: 'bg-green-50/50', path: '/retina' },
    { name: 'PeakProphet', description: 'Automated compound-peak matching from raw LC-MS and reaction data.', image: peakProphetLogo, color: 'text-purple-600', bg: 'bg-purple-50/50', path: '/peak-prophet' },
    { name: 'Gradience', description: 'Automated LC-MS gradient optimization.', image: gradienceLogo, color: 'text-red-500', bg: 'bg-red-50/50', path: '/gradience' }
  ];

  return (
    <div className="overflow-hidden bg-slate-50 min-h-screen relative font-sans">

      {/* Hero Section */}
      <section className="relative w-full min-h-[90vh] flex flex-col items-center pt-24 pb-0 overflow-hidden">

        {/* Graph Node Background */}
        <PeripheralGraph />

        {/* Content Container */}
        <div className="relative z-10 w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col items-center text-center h-full flex-grow">

          {/* Headline & Subheadline */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="max-w-4xl mx-auto mb-6"
          >
            <h1 className="text-5xl md:text-6xl font-extrabold tracking-tight text-slate-900 mb-4 drop-shadow-sm leading-tight">
              AI Driven <br />
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600">
                LC-MS Chemistry
              </span>
            </h1>
            <p className="mt-4 text-lg text-slate-600 max-w-2xl mx-auto leading-relaxed font-light">
              Integrate synthesis planning, property prediction, and method optimization in one unified platform.
            </p>
          </motion.div>

          {/* CTA Button */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="flex flex-col items-center justify-center gap-4 mb-8 lg:mb-12 z-20"
          >
            <a
              href="#modules"
              className="group inline-flex items-center justify-center px-8 py-3 text-base font-bold text-white bg-blue-600 rounded-full shadow-xl shadow-blue-500/30 hover:bg-blue-700 hover:scale-105 transition-all duration-200"
            >
              Get Started
            </a>


          </motion.div>

          {/* Central Chromatogram Animation */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1, delay: 0.4 }}
            className="w-full max-w-5xl relative mt-20 md:mt-auto"
          >
            <ChromatogramAnimation />
          </motion.div>

        </div>

        {/* Fade Transition to Modules */}
        <div className="absolute bottom-0 left-0 w-full h-32 bg-gradient-to-b from-transparent to-slate-50 pointer-events-none z-20"></div>
      </section>

      {/* Modules Section */}
      <section id="modules" className="py-16 relative bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 scroll-mt-20 overflow-hidden">
        <InteractiveBackground />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-extrabold text-slate-900 tracking-tight">
              Research Modules
            </h2>
            <p className="mt-4 text-lg text-slate-600 max-w-3xl mx-auto font-light leading-relaxed">
              Five tools working in harmony to accelerate your LC-MS research workflow.
            </p>
          </div>

          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-50px" }}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
          >
            {modules.map((module) => (
              <motion.div
                key={module.name}
                variants={itemVariants}
                className="group relative bg-white rounded-[1.5rem] border border-slate-200 p-6 shadow-sm hover:shadow-2xl hover:shadow-blue-900/20 transition-all duration-300 hover:-translate-y-2 min-h-[260px] flex flex-col justify-between"
              >
                <div>
                  <div className={`w-12 h-12 rounded-xl ${module.bg} flex items-center justify-center mb-6 transition-transform group-hover:scale-110 duration-300 overflow-hidden`}>
                    {module.image ? (
                      <img src={module.image} alt={`${module.name} logo`} className="w-full h-full object-cover" />
                    ) : (
                      <div className={`w-6 h-6 ${module.color}`} />
                    )}
                  </div>

                  <h3 className="text-xl font-bold text-slate-900 mb-3">
                    {module.name}
                  </h3>

                  <p className="text-base text-slate-600 leading-relaxed font-light">
                    {module.description}
                  </p>
                </div>

                <div className="mt-6 pt-6 border-t border-slate-100">
                  <Link
                    to={module.path}
                    className="inline-flex items-center text-sm font-bold text-blue-600 group-hover:text-blue-700"
                  >
                    Launch Module <ChevronRight className="w-4 h-4 ml-1 group-hover:translate-x-1 transition-transform duration-200" />
                  </Link>
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default Landing;
