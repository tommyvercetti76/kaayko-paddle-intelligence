#!/bin/bash
# Kaayko Data Collection Quick Start

echo "🌊 Kaayko Data Collection System"
echo "================================"
echo
echo "Available collection methods:"
echo "1. Interactive Menu:    python3 scripts/interactive_collector.py"
echo "2. HydroLAKES Direct:   python3 scripts/hydrolakes_collector.py --source /path/to/HydroLAKES_polys_v10.gdb --collect-weather"
echo
echo "Setup Requirements:"
echo "• WeatherAPI Key:       export KAAYKO_API_KEY='your_key_here'"
echo "• HydroLAKES Data:      Download from https://www.hydrosheds.org/products/hydrolakes"
echo
echo "Output directories:"
echo "• Weather Data:         ./output/collected_data/"
echo "• Coordinates:          ./output/coordinates/"
echo "• Logs:                 ./output/logs/"
echo
echo "Configuration:"
echo "• Copy .env.example to .env and add your API key"
echo "• Customize date ranges and thread count in .env"
