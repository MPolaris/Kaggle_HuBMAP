python .\train_args.py -filepath configs\config_resnest50fpn_se_20e_specifymeanstd
python .\train_args.py -filepath configs\config_resnest50fpn_se_20e
@REM python .\train_args.py -filepath configs\config_resnest50fpn
@REM python .\train_args.py -filepath configs\config_resnest50fpn_3x

@REM python .\submit_args.py -filepath configs\config_resnest50fpn_se_20e_specifymeanstd
@REM python .\submit_args.py -filepath configs\config_resnest50fpn_se_20e
@REM python .\submit_args.py -filepath configs\config_resnest50fpn
@REM python .\submit_args.py -filepath configs\config_resnest50fpn_3x

@REM python .\train_args.py -filepath 5foldconfigs\config_5fold_0
@REM python .\train_args.py -filepath 5foldconfigs\config_5fold_1
@REM python .\train_args.py -filepath 5foldconfigs\config_5fold_2
@REM python .\train_args.py -filepath 5foldconfigs\config_5fold_3
@REM python .\train_args.py -filepath 5foldconfigs\config_5fold_4

@REM python .\submit_args.py -filepath 5foldconfigs\config_5fold_0
@REM python .\submit_args.py -filepath 5foldconfigs\config_5fold_1
@REM python .\submit_args.py -filepath 5foldconfigs\config_5fold_2
@REM python .\submit_args.py -filepath 5foldconfigs\config_5fold_3
@REM python .\submit_args.py -filepath 5foldconfigs\config_5fold_4
@REM shutdown -s -t 60