import React from 'react';
import { Github, Mail } from 'lucide-react';

const Footer: React.FC = () => {
  return (
    <footer className="bg-slate-900 border-t border-slate-800 py-12 text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">

          <div className="col-span-1 md:col-span-2">
            <h3 className="text-xl font-bold text-white mb-4">
              LCOracle.ai
            </h3>
            <p className="text-slate-400 max-w-sm leading-relaxed">
              Integrating state-of-the-art machine learning models for LC-MS method development, product prediction, chemical property prediction, and gradient optimization.
            </p>
          </div>

          <div>
            <h4 className="font-bold text-white mb-4 tracking-wide uppercase text-sm">Modules</h4>
            <ul className="space-y-2 text-sm text-slate-400">
              <li><a href="/askcos" className="hover:text-white transition-colors">AskCOS</a></li>
              <li><a href="/amax" className="hover:text-white transition-colors">AMAX</a></li>
              <li><a href="/retina" className="hover:text-white transition-colors">ReTiNA</a></li>
              <li><a href="/peak-prophet" className="hover:text-white transition-colors">PeakProphet</a></li>
              <li><a href="/gradience" className="hover:text-white transition-colors">Gradience</a></li>
            </ul>
          </div>

          <div>
            <h4 className="font-bold text-white mb-4 tracking-wide uppercase text-sm">Connect</h4>
            <div className="flex space-x-4">
              <a href="https://github.com/natelgrw/lcoracle_ai" className="text-slate-400 hover:text-white transition-colors">
                <Github className="w-5 h-5" />
              </a>
              <a href="mailto:natelgrw.tech@gmail.com" className="text-slate-400 hover:text-white transition-colors">
                <Mail className="w-5 h-5" />
              </a>
            </div>
            <p className="mt-6 text-xs text-slate-500">
              v1.0.1 <br /><br /> Developed in collaboration with the Coley Research Group @ MIT
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
