#!/usr/bin/env python3
"""
🌊 KAAYKO HydroLAKES Interactive Weather Collection System
Beautiful menu-driven interface for comprehensive lake weather data collection
"""

import os
import sys
import time
from datetime import date, datetime
from pathlib import Path

# Import our main collection functions
try:
    from hydrolakes_collector import (
        load_hydrolakes_data, collect_weather_for_lake, 
        get_weather_api_key, save_lakes_to_csv, check_network_connectivity,
        SHOULD_STOP, OUTPUT_DIR, USA_COUNTRIES, INDIA_COUNTRIES, MIN_LAKE_AREA
    )
    from utils_ansi import C
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Please ensure you're running from the correct directory with all required files.")
    sys.exit(1)

# ASCII Art and UI Components
LOGO = f"""
{C.CYA}╭══════════════════════════════════════════════════════════════════════════════╮{C.R}
{C.CYA}│{C.R}{C.B}                     🌊 KAAYKO WEATHER COLLECTOR 🌊                      {C.R}{C.CYA}│{C.R}
{C.CYA}│{C.R}                      HydroLAKES Interactive System                       {C.CYA}│{C.R}
{C.CYA}╰══════════════════════════════════════════════════════════════════════════════╯{C.R}
"""

MENU_HEADER = f"""
{C.B}═══════════════════════════════════════════════════════════════════════════════{C.R}
{C.CYA}                            📋 COLLECTION MENU                                {C.R}
{C.B}═══════════════════════════════════════════════════════════════════════════════{C.R}
"""

def print_logo():
    """Display the beautiful logo."""
    os.system('clear' if os.name == 'posix' else 'cls')
    print(LOGO)

def print_separator():
    """Print a beautiful separator."""
    print(f"{C.GRY}{'─' * 80}{C.R}")

def get_user_input(prompt, input_type=str, default=None, choices=None):
    """Get validated user input with proper error handling."""
    while True:
        try:
            if default:
                display_prompt = f"{C.CYA}{prompt}{C.R} {C.GRY}(default: {default}){C.R}: "
            else:
                display_prompt = f"{C.CYA}{prompt}{C.R}: "
            
            user_input = input(display_prompt).strip()
            
            if not user_input and default is not None:
                return default
                
            if choices and user_input not in choices:
                print(f"{C.RED}❌ Invalid choice. Please select from: {', '.join(choices)}{C.R}")
                continue
                
            if input_type == int:
                return int(user_input)
            elif input_type == float:
                return float(user_input)
            else:
                return user_input
                
        except ValueError:
            print(f"{C.RED}❌ Invalid input type. Please enter a valid {input_type.__name__}.{C.R}")
        except KeyboardInterrupt:
            print(f"\n{C.YEL}👋 Collection cancelled by user{C.R}")
            sys.exit(0)

def show_main_menu():
    """Display the main collection menu."""
    print(MENU_HEADER)
    print(f"{C.GRN}🌍  1.{C.R} {C.B}Complete Global Collection{C.R}     - All lakes worldwide")
    print(f"{C.GRN}🇺🇸  2.{C.R} {C.B}USA Lakes Only{C.R}                - United States lakes")
    print(f"{C.GRN}🇮🇳  3.{C.R} {C.B}India Lakes Only{C.R}              - Indian subcontinent lakes")
    print(f"{C.GRN}🌎  4.{C.R} {C.B}USA + India Combined{C.R}           - Both countries")
    print(f"{C.GRN}🎯  5.{C.R} {C.B}Custom Region Selection{C.R}        - Choose specific countries")
    print(f"{C.GRN}🔍  6.{C.R} {C.B}Test Collection{C.R}                - Small sample for testing")
    print(f"{C.GRN}📊  7.{C.R} {C.B}Lake Coordinates Only{C.R}          - Extract coordinates without weather")
    print(f"{C.GRN}📈  8.{C.R} {C.B}Collection Status{C.R}              - View current progress")
    print(f"{C.GRN}⚙️   9.{C.R} {C.B}System Configuration{C.R}           - Setup and settings")
    print(f"{C.RED}❌  0.{C.R} {C.B}Exit{C.R}                           - Quit the system")
    print_separator()

def show_date_menu():
    """Get date range from user."""
    print(f"\n{C.CYA}📅 DATE RANGE CONFIGURATION{C.R}")
    print_separator()
    
    # Preset options
    today = date(2025, 8, 30)
    presets = {
        '1': ('2022-01-01', '2025-08-30', 'Full Range (2022 to present)'),
        '2': ('2023-01-01', '2025-08-30', 'Last 2+ Years'),
        '3': ('2024-01-01', '2025-08-30', 'Last Year'),
        '4': ('2024-08-01', '2025-08-30', 'Last Month'),
        '5': ('custom', 'custom', 'Custom Date Range')
    }
    
    print(f"{C.GRN}Date Range Options:{C.R}")
    for key, (start, end, desc) in presets.items():
        print(f"  {C.GRN}{key}.{C.R} {desc}")
        if start != 'custom':
            print(f"     {C.GRY}{start} to {end}{C.R}")
    
    choice = get_user_input("Select date range", choices=list(presets.keys()))
    
    if choice == '5':  # Custom
        start_date = get_user_input("Start date (YYYY-MM-DD)", default="2022-01-01")
        end_date = get_user_input("End date (YYYY-MM-DD)", default="2025-08-30")
        
        # Validate dates
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            print(f"{C.RED}❌ Invalid date format. Using default range.{C.R}")
            start_date, end_date = "2022-01-01", "2025-08-30"
    else:
        start_date, end_date, _ = presets[choice]
    
    return start_date, end_date

def show_collection_progress():
    """Show current collection status."""
    print_logo()
    print(f"{C.CYA}📊 COLLECTION STATUS{C.R}")
    print_separator()
    
    if not os.path.exists("weather_data"):
        print(f"{C.YEL}No collection data found yet.{C.R}")
        return
    
    # Count lakes and files
    lake_dirs = [d for d in os.listdir("weather_data") if os.path.isdir(f"weather_data/{d}")]
    total_files = 0
    
    for lake_dir in lake_dirs:
        lake_path = f"weather_data/{lake_dir}"
        csv_files = [f for f in os.listdir(lake_path) if f.endswith('.csv')]
        total_files += len(csv_files)
    
    print(f"{C.GRN}📁 Lake Directories:{C.R} {len(lake_dirs):,}")
    print(f"{C.GRN}📄 Weather Files:{C.R} {total_files:,}")
    
    # Show sample lakes
    if lake_dirs:
        print(f"\n{C.CYA}🌊 Sample Lakes:{C.R}")
        for i, lake in enumerate(sorted(lake_dirs)[:10], 1):
            lake_path = f"weather_data/{lake}"
            csv_count = len([f for f in os.listdir(lake_path) if f.endswith('.csv')])
            print(f"  {i:2d}. {lake:<30} ({csv_count} months)")
        
        if len(lake_dirs) > 10:
            print(f"  ... and {len(lake_dirs) - 10} more lakes")
    
    print_separator()
    input(f"{C.GRY}Press Enter to continue...{C.R}")

def show_system_config():
    """Show and configure system settings."""
    print_logo()
    print(f"{C.CYA}⚙️ SYSTEM CONFIGURATION{C.R}")
    print_separator()
    
    # Show current configuration
    api_key = os.getenv("WEATHERAPI_KEY", "")
    output_dir = os.getenv("KAAYKO_OUTPUT_DIR", OUTPUT_DIR)
    
    print(f"{C.GRN}Current Settings:{C.R}")
    print(f"  API Key: {C.GRY}{'Set' if api_key else 'Not set'}{C.R}")
    print(f"  Output Directory: {C.GRY}{output_dir}{C.R}")
    print(f"  Minimum Lake Area: {C.GRY}{MIN_LAKE_AREA} km²{C.R}")
    
    print(f"\n{C.CYA}Configuration Options:{C.R}")
    print(f"  {C.GRN}1.{C.R} Set WeatherAPI Key")
    print(f"  {C.GRN}2.{C.R} Change Output Directory") 
    print(f"  {C.GRN}3.{C.R} Test Network Connection")
    print(f"  {C.GRN}0.{C.R} Back to Main Menu")
    
    choice = get_user_input("Select option", choices=['0', '1', '2', '3'])
    
    if choice == '1':
        new_key = get_user_input("Enter WeatherAPI key")
        os.environ["WEATHERAPI_KEY"] = new_key
        print(f"{C.GRN}✅ API key updated{C.R}")
        
    elif choice == '2':
        new_dir = get_user_input("Enter output directory", default=output_dir)
        os.environ["KAAYKO_OUTPUT_DIR"] = new_dir
        print(f"{C.GRN}✅ Output directory updated to: {new_dir}{C.R}")
        
    elif choice == '3':
        print(f"{C.GRY}Testing network connection...{C.R}")
        if check_network_connectivity():
            print(f"{C.GRN}✅ Network connection successful{C.R}")
        else:
            print(f"{C.RED}❌ Network connection failed{C.R}")
    
    if choice != '0':
        input(f"\n{C.GRY}Press Enter to continue...{C.R}")

def run_collection(collection_config):
    """Execute the weather collection with given configuration."""
    print_logo()
    print(f"{C.CYA}🚀 STARTING COLLECTION{C.R}")
    print_separator()
    
    # Display collection summary
    print(f"{C.GRN}Collection Configuration:{C.R}")
    print(f"  Target: {C.B}{collection_config['description']}{C.R}")
    print(f"  Date Range: {C.B}{collection_config['start_date']} to {collection_config['end_date']}{C.R}")
    if 'limit' in collection_config:
        print(f"  Lake Limit: {C.B}{collection_config['limit']:,}{C.R}")
    
    # Confirm with user
    confirm = get_user_input(f"\n{C.YEL}Proceed with collection? (y/n){C.R}", choices=['y', 'n'])
    if confirm != 'y':
        print(f"{C.YEL}Collection cancelled{C.R}")
        return
    
    try:
        # Get API key
        api_key = get_weather_api_key()
        
        # Load lakes
        print(f"\n{C.BLU}Loading HydroLAKES data...{C.R}")
        lakes = load_hydrolakes_data(
            limit=collection_config.get('limit'),
            per_country=collection_config.get('per_country', 0),
            usa_lakes=collection_config.get('usa_lakes'),
            india_lakes=collection_config.get('india_lakes')
        )
        
        if not lakes:
            print(f"{C.RED}❌ No lakes found matching criteria{C.R}")
            return
        
        # Save coordinates if requested
        if collection_config.get('save_coords'):
            coords_file = f"coordinates_{collection_config['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            save_lakes_to_csv(lakes, coords_file)
            print(f"{C.GRN}✅ Lake coordinates saved to: {coords_file}{C.R}")
        
        # Weather collection
        if collection_config.get('collect_weather', True):
            print(f"\n{C.BLU}Starting weather data collection...{C.R}")
            print(f"{C.GRY}Processing {len(lakes):,} lakes...{C.R}")
            
            total_rows = 0
            total_months = 0
            processed = 0
            failed_lakes = 0
            
            start_time = time.time()
            
            for i, lake in enumerate(lakes, 1):
                if SHOULD_STOP.is_set():
                    print(f"\n{C.YEL}Collection interrupted by user{C.R}")
                    break
                
                print(f"\n{C.CYA}[{i:,}/{len(lakes):,}]{C.R} {lake['lake_name']} ({lake['country']})")
                
                rows, months = collect_weather_for_lake(
                    lake, 
                    collection_config['start_date'], 
                    collection_config['end_date'], 
                    api_key
                )
                
                if rows == 0 and months == 0:
                    failed_lakes += 1
                
                total_rows += rows
                total_months += months
                processed += 1
                
                # Progress update every 10 lakes
                if processed % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / processed
                    remaining = (len(lakes) - processed) * avg_time
                    
                    print(f"\n{C.GRY}Progress Update:{C.R}")
                    print(f"  Processed: {processed}/{len(lakes)} lakes")
                    print(f"  Records: {total_rows:,} weather entries")
                    print(f"  Failed: {failed_lakes} lakes")
                    print(f"  ETA: {remaining/60:.1f} minutes")
            
            # Final summary
            print(f"\n{C.GRN}🎉 COLLECTION COMPLETE{C.R}")
            print(f"{C.GRN}Statistics:{C.R}")
            print(f"  • Processed: {processed:,} lakes")
            print(f"  • Weather records: {total_rows:,}")
            print(f"  • Months collected: {total_months:,}")
            print(f"  • Failed: {failed_lakes:,} lakes")
            print(f"  • Total time: {(time.time() - start_time)/60:.1f} minutes")
            print(f"  • Output directory: {OUTPUT_DIR}")
        
    except KeyboardInterrupt:
        print(f"\n{C.YEL}Collection interrupted by user{C.R}")
        SHOULD_STOP.set()
    except Exception as e:
        print(f"\n{C.RED}❌ Collection failed: {e}{C.R}")
        import traceback
        traceback.print_exc()
    
    input(f"\n{C.GRY}Press Enter to continue...{C.R}")

def main():
    """Main interactive loop."""
    print_logo()
    
    # Check initial setup
    if not check_network_connectivity():
        print(f"{C.RED}❌ Network connectivity issue. Please check your connection.{C.R}")
        input("Press Enter to continue anyway...")
    
    while True:
        print_logo()
        show_main_menu()
        
        choice = get_user_input("Select an option", choices=[str(i) for i in range(10)])
        
        if choice == '0':  # Exit
            print(f"\n{C.GRN}👋 Thank you for using KAAYKO Weather Collector!{C.R}")
            sys.exit(0)
        
        elif choice == '1':  # Complete Global Collection
            start_date, end_date = show_date_menu()
            config = {
                'name': 'global_complete',
                'description': 'All lakes worldwide',
                'start_date': start_date,
                'end_date': end_date,
                'save_coords': True,
                'collect_weather': True
            }
            run_collection(config)
        
        elif choice == '2':  # USA Only
            start_date, end_date = show_date_menu()
            config = {
                'name': 'usa_only',
                'description': 'USA lakes only',
                'start_date': start_date,
                'end_date': end_date,
                'usa_lakes': None,  # All USA lakes
                'save_coords': True,
                'collect_weather': True
            }
            run_collection(config)
        
        elif choice == '3':  # India Only
            start_date, end_date = show_date_menu()
            config = {
                'name': 'india_only',
                'description': 'India lakes only',
                'start_date': start_date,
                'end_date': end_date,
                'india_lakes': None,  # All India lakes
                'save_coords': True,
                'collect_weather': True
            }
            run_collection(config)
        
        elif choice == '4':  # USA + India Combined
            start_date, end_date = show_date_menu()
            config = {
                'name': 'usa_india_combined',
                'description': 'USA + India lakes combined',
                'start_date': start_date,
                'end_date': end_date,
                'usa_lakes': None,
                'india_lakes': None,
                'save_coords': True,
                'collect_weather': True
            }
            run_collection(config)
        
        elif choice == '5':  # Custom Region
            print(f"\n{C.YEL}Custom region selection not yet implemented{C.R}")
            input("Press Enter to continue...")
        
        elif choice == '6':  # Test Collection
            print(f"\n{C.CYA}🧪 TEST COLLECTION SETUP{C.R}")
            limit = get_user_input("Number of lakes to test", input_type=int, default=5)
            start_date, end_date = show_date_menu()
            
            config = {
                'name': 'test_collection',
                'description': f'Test collection ({limit} lakes)',
                'start_date': start_date,
                'end_date': end_date,
                'limit': limit,
                'save_coords': True,
                'collect_weather': True
            }
            run_collection(config)
        
        elif choice == '7':  # Coordinates Only
            print(f"\n{C.CYA}📊 COORDINATES EXTRACTION{C.R}")
            limit = get_user_input("Lake limit (0 for all)", input_type=int, default=0)
            
            config = {
                'name': 'coordinates_only',
                'description': 'Lake coordinates extraction',
                'start_date': '2022-01-01',
                'end_date': '2025-08-30',
                'limit': limit if limit > 0 else None,
                'save_coords': True,
                'collect_weather': False
            }
            run_collection(config)
        
        elif choice == '8':  # Collection Status
            show_collection_progress()
        
        elif choice == '9':  # System Configuration
            show_system_config()

if __name__ == "__main__":
    main()
