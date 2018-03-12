image_input_path = 'C:\Users\Yola\AppData\Local\Programs\Python\Python36\Scripts\tensor_test\input_images\*.png';
output_image_path = 'C:\Users\Yola\AppData\Local\Programs\Python\Python36\Scripts\tensor_test\training_data\image_';

image_size = 28;

imagefiles = dir(image_input_path);   
nfiles = length(imagefiles);    % Number of files found
for i=1:nfiles
   currentfilename = imagefiles(i).name;
   currentfilename = strcat('input_images\',currentfilename);
   currentimage = imread(currentfilename);
   images{i} = currentimage;
   [~,~,size_z] = size(images{i});
   if(size_z > 1)
     images{i} = rgb2gray(images{i});  
   end
   %figure;
   %imshow(images{i});
   %figure;
   [size_y,size_x,size_z] = size(images{i});
   for j=1:20
       x_start = randi([1,(size_x-image_size)],1,1);
       y_start = randi([1,(size_y-image_size)],1,1);
       new_image = zeros(image_size);
       new_image = images{i}(y_start:y_start+(image_size-1), x_start:x_start+(image_size-1));
       blurred_image = imgaussfilt(new_image,1);
       total_image = zeros(image_size,2*image_size);
       for image_inner=1:image_size
           total_image(:,image_inner)=blurred_image(:,image_inner);
           total_image(:,(image_inner)+image_size)=new_image(:,image_inner);
       end
       total_image = mat2gray(total_image);
       tmp = int2str((((i-1)*20)+j));
       image_path = strcat(output_image_path, tmp, '.png');
       imwrite(total_image, image_path);
   end
end

